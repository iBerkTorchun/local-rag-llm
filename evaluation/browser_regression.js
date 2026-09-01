"use strict";

const fs = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");
const outputPath = path.join(__dirname, "final_browser_regression_verified.json");
const debugBase = process.env.CHROME_DEBUG_URL || "http://127.0.0.1:9222";
const appUrl = "http://127.0.0.1:5000/";

if (fs.existsSync(outputPath)) {
  throw new Error(`Refusing to overwrite ${outputPath}`);
}

const delay = (milliseconds) => new Promise((resolve) => {
  setTimeout(resolve, milliseconds);
});

async function waitFor(check, description, timeoutMilliseconds = 30000) {
  const deadline = Date.now() + timeoutMilliseconds;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) {
      return value;
    }
    await delay(100);
  }
  throw new Error(`Timed out waiting for ${description}.`);
}

async function main() {
  const targets = await fetch(`${debugBase}/json/list`).then((response) => response.json());
  const page = targets.find((target) => target.type === "page");
  if (!page) {
    throw new Error("No Chrome page target was available.");
  }

  const socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  let nextId = 1;
  const pending = new Map();
  const exceptions = [];
  const consoleErrors = [];
  const askRequests = [];

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) {
        reject(new Error(JSON.stringify(message.error)));
      } else {
        resolve(message.result);
      }
      return;
    }

    if (message.method === "Runtime.exceptionThrown") {
      exceptions.push(message.params.exceptionDetails.text);
    }
    if (
      message.method === "Runtime.consoleAPICalled"
      && message.params.type === "error"
    ) {
      consoleErrors.push(message.params.args.map((argument) => argument.value).join(" "));
    }
    if (
      message.method === "Network.requestWillBeSent"
      && message.params.request.url.endsWith("/api/ask")
    ) {
      askRequests.push({
        url: message.params.request.url,
        method: message.params.request.method,
        postData: message.params.request.postData,
      });
    }
  });

  function command(method, params = {}) {
    const id = nextId;
    nextId += 1;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params }));
    });
  }

  async function evaluate(expression) {
    const result = await command("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text);
    }
    return result.result.value;
  }

  try {
    await command("Runtime.enable");
    await command("Page.enable");
    await command("Network.enable");
    await command("Emulation.setDeviceMetricsOverride", {
      width: 1280,
      height: 900,
      deviceScaleFactor: 1,
      mobile: false,
    });
    await command("Page.navigate", { url: appUrl });
    await waitFor(
      () => evaluate("document.readyState === 'complete'"),
      "page load",
    );
    await waitFor(
      () => evaluate("document.querySelector('#api-status-text')?.textContent === 'Local API available'"),
      "health status",
    );

    const initial = await evaluate(`(() => ({
      title: document.title,
      health: document.querySelector('#api-status-text').textContent,
      emptyStateVisible: !document.querySelector('#empty-state').hidden,
      transcriptHidden: document.querySelector('#transcript').hidden,
      theme: document.documentElement.dataset.theme,
      composerPosition: getComputedStyle(document.querySelector('#composer-section')).position,
      horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth
    }))()`);

    async function submitQuestion(question, expectedTurns) {
      const startedAt = performance.now();
      await evaluate(`(() => {
        const input = document.querySelector('#question-input');
        input.value = ${JSON.stringify(question)};
        input.dispatchEvent(new Event('input', { bubbles: true }));
        document.querySelector('#question-form').requestSubmit();
        return true;
      })()`);
      await waitFor(
        () => evaluate(`document.querySelectorAll('.conversation-turn').length === ${expectedTurns}
          && document.querySelectorAll('.assistant-message:not(.assistant-pending)').length === ${expectedTurns}`),
        `assistant response ${expectedTurns}`,
      );
      return (performance.now() - startedAt) / 1000;
    }

    const questions = [
      "Why is SQLite suitable for this local project?",
      "What role does Foundry Local play in the application?",
      "How many vacation days do users of this application receive?",
    ];
    const browserTimings = [];
    browserTimings.push(await submitQuestion(questions[0], 1));

    const collapsed = await evaluate(`(() => {
      const button = document.querySelector('.source-toggle');
      const region = document.querySelector('.source-details');
      return {
        buttonText: button.textContent.trim(),
        expanded: button.getAttribute('aria-expanded'),
        regionHidden: region.hidden
      };
    })()`);
    await evaluate("document.querySelector('.source-toggle').click(); true");
    const expanded = await evaluate(`(() => {
      const button = document.querySelector('.source-toggle');
      const region = document.querySelector('.source-details');
      return {
        expanded: button.getAttribute('aria-expanded'),
        regionHidden: region.hidden,
        sourceCount: region.querySelectorAll('.source-list > li').length,
        sources: [...region.querySelectorAll('.source-list > li')].map((item) => ({
          source: item.querySelector('.source-title').textContent,
          metadata: item.querySelector('.source-meta').textContent,
          contentPresent: item.querySelector('.source-content').textContent.trim().length > 0
        }))
      };
    })()`);
    const desktopAnswered = await evaluate(`(() => ({
      composerPosition: getComputedStyle(document.querySelector('#composer-section')).position,
      horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth
    }))()`);

    const themeBefore = await evaluate("document.documentElement.dataset.theme");
    await evaluate("document.querySelector('#theme-toggle').click(); true");
    const themeAfter = await evaluate(`(() => ({
      theme: document.documentElement.dataset.theme,
      saved: localStorage.getItem('local-knowledge-theme'),
      turns: document.querySelectorAll('.conversation-turn').length
    }))()`);

    browserTimings.push(await submitQuestion(questions[1], 2));
    browserTimings.push(await submitQuestion(questions[2], 3));
    const transcript = await evaluate(`(() => ({
      turns: document.querySelectorAll('.conversation-turn').length,
      userLabels: [...document.querySelectorAll('.user-message .message-label')].map((node) => node.textContent),
      assistantLabels: [...document.querySelectorAll('.assistant-message .message-label')].map((node) => node.textContent),
      questions: [...document.querySelectorAll('.user-text')].map((node) => node.textContent),
      answers: [...document.querySelectorAll('.assistant-answer')].map((node) => node.textContent),
      sourceButtons: document.querySelectorAll('.source-toggle').length,
      technicalErrors: document.querySelectorAll('.technical-error').length,
      theme: document.documentElement.dataset.theme
    }))()`);

    await command("Emulation.setDeviceMetricsOverride", {
      width: 390,
      height: 844,
      deviceScaleFactor: 1,
      mobile: true,
    });
    const mobile = await evaluate(`(() => {
      const composer = document.querySelector('#composer-section').getBoundingClientRect();
      const expandedRegion = document.querySelector('.source-details:not([hidden])');
      return {
        viewport: { width: window.innerWidth, height: window.innerHeight },
        horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
        composerPosition: getComputedStyle(document.querySelector('#composer-section')).position,
        composerWidth: composer.width,
        expandedSourceReadable: expandedRegion ? expandedRegion.scrollWidth <= expandedRegion.clientWidth : false,
        theme: document.documentElement.dataset.theme
      };
    })()`);

    const parsedRequests = askRequests.map((request) => ({
      method: request.method,
      body: JSON.parse(request.postData),
    }));
    const independentQuestionsOnly = parsedRequests.length === questions.length
      && parsedRequests.every((request, index) => (
        request.method === "POST"
        && Object.keys(request.body).length === 1
        && request.body.question === questions[index]
      ));

    const checks = {
      initialState: initial.emptyStateVisible && initial.transcriptHidden,
      healthStatus: initial.health === "Local API available",
      stickyComposer: desktopAnswered.composerPosition === "sticky"
        && mobile.composerPosition === "sticky",
      sourcesCollapsedInitially: collapsed.expanded === "false" && collapsed.regionHidden,
      sourceDisclosure: expanded.expanded === "true"
        && !expanded.regionHidden
        && expanded.sourceCount === 3
        && expanded.sources.every((source) => source.contentPresent),
      themeTogglePreservesTranscript: themeBefore !== themeAfter.theme
        && themeAfter.saved === themeAfter.theme
        && themeAfter.turns === 1,
      sessionTranscript: transcript.turns === 3
        && transcript.sourceButtons === 3
        && transcript.technicalErrors === 0,
      groundedFallbackNormalAnswer: transcript.answers[2]
        === "The information is not available in the supplied context.",
      independentApiRequests: independentQuestionsOnly,
      desktopNoOverflow: !initial.horizontalOverflow
        && !desktopAnswered.horizontalOverflow,
      mobileNoOverflow: !mobile.horizontalOverflow && mobile.expandedSourceReadable,
      noConsoleBreakingErrors: exceptions.length === 0 && consoleErrors.length === 0,
    };

    const payload = {
      generatedAtUtc: new Date().toISOString(),
      browser: "Google Chrome headless via built-in DevTools Protocol",
      appUrl,
      initial,
      desktopAnswered,
      collapsedSources: collapsed,
      expandedSources: expanded,
      themeBefore,
      themeAfter,
      transcript,
      mobile,
      browserRequestSeconds: browserTimings.map((value) => Number(value.toFixed(3))),
      apiRequests: parsedRequests,
      exceptions,
      consoleErrors,
      checks,
      allChecksPassed: Object.values(checks).every(Boolean),
    };
    fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
    console.log(JSON.stringify({ outputPath, checks, browserRequestSeconds: payload.browserRequestSeconds }, null, 2));
    if (!payload.allChecksPassed) {
      process.exitCode = 1;
    }
  } finally {
    socket.close();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});

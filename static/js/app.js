"use strict";

const form = document.querySelector("#question-form");
const composerSection = document.querySelector("#composer-section");
const questionInput = document.querySelector("#question-input");
const submitButton = document.querySelector("#submit-button");
const validationMessage = document.querySelector("#question-validation");
const emptyState = document.querySelector("#empty-state");
const transcript = document.querySelector("#transcript");
const apiStatus = document.querySelector("#api-status");
const apiStatusText = document.querySelector("#api-status-text");
const themeToggle = document.querySelector("#theme-toggle");
const liveStatus = document.querySelector("#live-status");

const GENERIC_ERROR = "The local request could not be completed. Please try again.";
const UNREADABLE_RESPONSE_ERROR =
  "The local API returned an unreadable response. Please try again.";
const THEME_STORAGE_KEY = "local-knowledge-theme";

const turns = [];
let nextTurnId = 1;
let isLoading = false;
let isComposing = false;

class DisplayError extends Error {}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

async function parseJson(response) {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.toLowerCase().includes("application/json")) {
    throw new DisplayError(UNREADABLE_RESPONSE_ERROR);
  }

  try {
    return await response.json();
  } catch {
    throw new DisplayError(UNREADABLE_RESPONSE_ERROR);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", {
      headers: { Accept: "application/json" },
    });
    const payload = await parseJson(response);

    if (!response.ok || !isObject(payload) || payload.status !== "ok") {
      throw new Error("Unexpected health response.");
    }

    apiStatus.dataset.state = "available";
    apiStatusText.textContent = "Local API available";
  } catch {
    apiStatus.dataset.state = "unavailable";
    apiStatusText.textContent = "Local API unavailable";
  }
}

function readStoredTheme() {
  try {
    const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    return savedTheme === "light" || savedTheme === "dark" ? savedTheme : null;
  } catch {
    return null;
  }
}

function updateThemeControl(theme) {
  const nextTheme = theme === "dark" ? "light" : "dark";
  const label = `Switch to ${nextTheme} theme`;
  themeToggle.setAttribute("aria-label", label);
  themeToggle.setAttribute("title", label);
  themeToggle.setAttribute("aria-pressed", String(theme === "dark"));
}

function applyTheme(theme, { persist = false, announce = false } = {}) {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;

  if (persist) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // The selected theme still applies for this page if storage is unavailable.
    }
  }

  updateThemeControl(theme);
  if (announce) {
    liveStatus.textContent = `${theme === "dark" ? "Dark" : "Light"} theme selected.`;
  }
}

function initializeTheme() {
  const currentTheme = document.documentElement.dataset.theme === "dark"
    ? "dark"
    : "light";
  applyTheme(currentTheme);

  const systemPreference = window.matchMedia("(prefers-color-scheme: dark)");
  const handleSystemChange = (event) => {
    if (readStoredTheme() === null) {
      applyTheme(event.matches ? "dark" : "light");
    }
  };

  if (typeof systemPreference.addEventListener === "function") {
    systemPreference.addEventListener("change", handleSystemChange);
  } else if (typeof systemPreference.addListener === "function") {
    systemPreference.addListener(handleSystemChange);
  }
}

function validateAnswerPayload(payload) {
  if (
    !isObject(payload) ||
    typeof payload.answer !== "string" ||
    payload.answer.trim() === "" ||
    !Array.isArray(payload.sources) ||
    payload.sources.length !== 3
  ) {
    return false;
  }

  return payload.sources.every(
    (source) =>
      isObject(source) &&
      typeof source.source === "string" &&
      typeof source.chunk_index === "number" &&
      Number.isFinite(source.chunk_index) &&
      typeof source.score === "number" &&
      Number.isFinite(source.score) &&
      typeof source.content === "string",
  );
}

function clearValidation() {
  validationMessage.textContent = "";
  validationMessage.hidden = true;
  questionInput.removeAttribute("aria-invalid");
}

function showValidation(message) {
  validationMessage.textContent = message;
  validationMessage.hidden = false;
  questionInput.setAttribute("aria-invalid", "true");
  questionInput.focus();
}

function setLoading(loading) {
  isLoading = loading;
  submitButton.disabled = loading;
  submitButton.textContent = loading ? "Working\u2026" : "Ask";
  questionInput.readOnly = loading;
  form.setAttribute("aria-busy", String(loading));
  transcript.setAttribute("aria-busy", String(loading));
}

function focusAndReveal(element) {
  element.focus({ preventScroll: true });
  const bounds = element.getBoundingClientRect();
  if (bounds.top < 16 || bounds.bottom > window.innerHeight - 16) {
    element.scrollIntoView({ block: "nearest", behavior: "auto" });
  }
}

function revealIfNeeded(element) {
  const bounds = element.getBoundingClientRect();
  if (bounds.top < 16 || bounds.bottom > window.innerHeight - 16) {
    element.scrollIntoView({ block: "nearest", behavior: "auto" });
  }
}

function createTurn(question) {
  const id = nextTurnId;
  nextTurnId += 1;

  const item = document.createElement("li");
  const article = document.createElement("article");
  const userMessage = document.createElement("section");
  const userLabel = document.createElement("h3");
  const userText = document.createElement("p");
  const responseSlot = document.createElement("div");

  item.className = "conversation-turn";
  article.setAttribute("aria-labelledby", `user-label-${id}`);
  userMessage.className = "user-message";
  userLabel.className = "message-label";
  userLabel.id = `user-label-${id}`;
  userLabel.textContent = "You";
  userText.className = "user-text";
  userText.textContent = question;
  responseSlot.className = "response-slot";

  userMessage.append(userLabel, userText);
  article.append(userMessage, responseSlot);
  item.append(article);
  transcript.append(item);

  emptyState.hidden = true;
  transcript.hidden = false;
  composerSection.classList.add("has-transcript");

  const turn = {
    id,
    question,
    answer: null,
    sources: [],
    status: "pending",
    element: item,
    responseSlot,
  };
  turns.push(turn);
  return turn;
}

function createMessageLabel(id, text) {
  const heading = document.createElement("h3");
  heading.className = "message-label";
  heading.id = id;
  heading.textContent = text;
  return heading;
}

function renderPending(turn) {
  const section = document.createElement("section");
  const heading = createMessageLabel(`assistant-label-${turn.id}`, "Assistant");
  const status = document.createElement("p");

  section.className = "assistant-message assistant-pending";
  section.setAttribute("aria-labelledby", heading.id);
  status.className = "processing-status";
  status.textContent = "Processing your question locally\u2026";

  section.append(heading, status);
  turn.responseSlot.replaceChildren(section);
  turn.element.setAttribute("aria-busy", "true");
  turn.status = "pending";
  return heading;
}

function createDocumentIcon() {
  const namespace = "http://www.w3.org/2000/svg";
  const icon = document.createElementNS(namespace, "svg");
  const path = document.createElementNS(namespace, "path");

  icon.setAttribute("aria-hidden", "true");
  icon.setAttribute("viewBox", "0 0 20 20");
  icon.setAttribute("focusable", "false");
  path.setAttribute("d", "M5.5 2.75h6l3 3v11.5h-9zM11.5 2.75v3h3M8 9h4M8 12h4");
  icon.append(path);
  return icon;
}

function renderSourceList(sources) {
  const list = document.createElement("ol");
  list.className = "source-list";

  sources.forEach((source) => {
    const item = document.createElement("li");
    const article = document.createElement("article");
    const title = document.createElement("h5");
    const metadata = document.createElement("p");
    const content = document.createElement("p");

    title.className = "source-title";
    title.textContent = source.source;
    metadata.className = "source-meta";
    metadata.textContent =
      `Chunk ${source.chunk_index} \u00b7 Similarity ${source.score.toFixed(4)}`;
    content.className = "source-content";
    content.textContent = source.content;

    article.append(title, metadata, content);
    item.append(article);
    list.append(item);
  });

  return list;
}

function renderAnswer(turn, payload) {
  const section = document.createElement("section");
  const heading = createMessageLabel(`assistant-label-${turn.id}`, "Assistant");
  const answer = document.createElement("div");
  const sourceButton = document.createElement("button");
  const sourceRegion = document.createElement("div");
  const sourceHeading = document.createElement("h4");
  const sourceButtonId = `source-toggle-${turn.id}`;
  const sourceRegionId = `source-details-${turn.id}`;

  section.className = "assistant-message";
  section.setAttribute("aria-labelledby", heading.id);
  heading.tabIndex = -1;
  answer.className = "assistant-answer";
  answer.textContent = payload.answer.trim();

  sourceButton.className = "source-toggle";
  sourceButton.id = sourceButtonId;
  sourceButton.type = "button";
  sourceButton.setAttribute("aria-expanded", "false");
  sourceButton.setAttribute("aria-controls", sourceRegionId);
  sourceButton.append(createDocumentIcon(), document.createTextNode(`${payload.sources.length} sources`));

  sourceRegion.className = "source-details";
  sourceRegion.id = sourceRegionId;
  sourceRegion.setAttribute("role", "region");
  sourceRegion.setAttribute("aria-labelledby", sourceButtonId);
  sourceRegion.hidden = true;
  sourceHeading.className = "visually-hidden";
  sourceHeading.textContent = "Sources for this answer";
  sourceRegion.append(sourceHeading, renderSourceList(payload.sources));

  sourceButton.addEventListener("click", () => {
    const shouldExpand = sourceButton.getAttribute("aria-expanded") !== "true";
    sourceButton.setAttribute("aria-expanded", String(shouldExpand));
    sourceRegion.hidden = !shouldExpand;
  });

  section.append(heading, answer, sourceButton, sourceRegion);
  turn.responseSlot.replaceChildren(section);
  turn.element.removeAttribute("aria-busy");
  turn.answer = payload.answer.trim();
  turn.sources = payload.sources.map((source) => ({ ...source }));
  turn.status = "complete";

  liveStatus.textContent = "Answer ready. Three supporting sources available.";
  focusAndReveal(heading);
}

function renderTechnicalError(turn, message) {
  const section = document.createElement("section");
  const heading = createMessageLabel(`error-label-${turn.id}`, "Request error");
  const detail = document.createElement("p");
  const retryButton = document.createElement("button");

  section.className = "technical-error";
  section.setAttribute("role", "alert");
  section.setAttribute("aria-labelledby", heading.id);
  heading.tabIndex = -1;
  detail.textContent = message;
  retryButton.className = "secondary-button";
  retryButton.type = "button";
  retryButton.textContent = "Retry";
  retryButton.addEventListener("click", () => {
    performRequest(turn, { fromRetry: true });
  });

  section.append(heading, detail, retryButton);
  turn.responseSlot.replaceChildren(section);
  turn.element.removeAttribute("aria-busy");
  turn.status = "error";
  liveStatus.textContent = "";
  focusAndReveal(heading);
}

async function requestAnswer(question) {
  let response;

  try {
    response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new DisplayError(
      "Could not reach the local API. Make sure the Flask server is running and try again.",
    );
  }

  const payload = await parseJson(response);

  if (!response.ok) {
    const backendMessage =
      isObject(payload) && typeof payload.error === "string"
        ? payload.error.trim()
        : "";
    throw new DisplayError(backendMessage || GENERIC_ERROR);
  }

  if (!validateAnswerPayload(payload)) {
    throw new DisplayError(
      "The local API returned an unexpected response. Please try again.",
    );
  }

  return payload;
}

async function performRequest(turn, { fromRetry = false } = {}) {
  if (isLoading) {
    return;
  }

  questionInput.value = turn.question;
  const pendingHeading = renderPending(turn);
  setLoading(true);
  liveStatus.textContent = "Processing your question locally.";

  if (fromRetry) {
    pendingHeading.tabIndex = -1;
    focusAndReveal(pendingHeading);
  } else {
    window.requestAnimationFrame(() => revealIfNeeded(turn.element));
  }

  try {
    const payload = await requestAnswer(turn.question);
    setLoading(false);
    questionInput.value = "";
    renderAnswer(turn, payload);
  } catch (error) {
    setLoading(false);
    const message = error instanceof DisplayError ? error.message : GENERIC_ERROR;
    renderTechnicalError(turn, message);
  }
}

function submitQuestion(rawQuestion) {
  if (isLoading) {
    return;
  }

  const question = rawQuestion.trim();
  if (!question) {
    showValidation("Enter a question before submitting.");
    return;
  }

  clearValidation();
  questionInput.value = question;
  const turn = createTurn(question);
  performRequest(turn);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(questionInput.value);
});

questionInput.addEventListener("compositionstart", () => {
  isComposing = true;
});

questionInput.addEventListener("compositionend", () => {
  isComposing = false;
});

questionInput.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing &&
    !isComposing
  ) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionInput.addEventListener("input", () => {
  if (questionInput.value.trim()) {
    clearValidation();
  }
});

themeToggle.addEventListener("click", () => {
  const currentTheme = document.documentElement.dataset.theme === "dark"
    ? "dark"
    : "light";
  applyTheme(currentTheme === "dark" ? "light" : "dark", {
    persist: true,
    announce: true,
  });
});

initializeTheme();
checkHealth();

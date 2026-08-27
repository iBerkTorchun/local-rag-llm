"use strict";

const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question-input");
const submitButton = document.querySelector("#submit-button");
const validationMessage = document.querySelector("#question-validation");
const apiStatus = document.querySelector("#api-status");
const apiStatusText = document.querySelector("#api-status-text");
const requestStatusSection = document.querySelector("#request-status-section");
const pendingQuestion = document.querySelector("#pending-question");
const resultSection = document.querySelector("#result-section");
const resultQuestion = document.querySelector("#result-question");
const answerHeading = document.querySelector("#answer-heading");
const answerText = document.querySelector("#answer-text");
const sourceList = document.querySelector("#source-list");
const errorSection = document.querySelector("#error-section");
const errorHeading = document.querySelector("#error-heading");
const errorMessage = document.querySelector("#error-message");
const retryButton = document.querySelector("#retry-button");
const liveStatus = document.querySelector("#live-status");

const GENERIC_ERROR = "The local request could not be completed. Please try again.";
const UNREADABLE_RESPONSE_ERROR =
  "The local API returned an unreadable response. Please try again.";

let isLoading = false;
let isComposing = false;
let lastQuestion = "";

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

function setLoading(loading, question = "") {
  isLoading = loading;
  submitButton.disabled = loading;
  submitButton.textContent = loading ? "Working…" : "Ask";
  questionInput.readOnly = loading;
  form.setAttribute("aria-busy", String(loading));
  requestStatusSection.setAttribute("aria-busy", String(loading));

  if (loading) {
    pendingQuestion.textContent = question;
    requestStatusSection.hidden = false;
    resultSection.hidden = true;
    errorSection.hidden = true;
    liveStatus.textContent = "Processing your question locally.";
  }
}

function renderSources(sources) {
  const fragment = document.createDocumentFragment();

  sources.forEach((source) => {
    const item = document.createElement("li");
    const article = document.createElement("article");
    const title = document.createElement("h3");
    const metadata = document.createElement("p");
    const content = document.createElement("p");

    title.className = "source-title";
    title.textContent = source.source;

    metadata.className = "source-meta";
    metadata.textContent =
      `Chunk ${source.chunk_index} · Similarity ${source.score.toFixed(4)}`;

    content.className = "source-content";
    content.textContent = source.content;

    article.append(title, metadata, content);
    item.append(article);
    fragment.append(item);
  });

  sourceList.replaceChildren(fragment);
}

function showResult(question, payload) {
  requestStatusSection.hidden = true;
  errorSection.hidden = true;
  resultQuestion.textContent = question;
  answerText.textContent = payload.answer.trim();
  renderSources(payload.sources);
  resultSection.hidden = false;
  liveStatus.textContent = "Answer ready. Three supporting sources retrieved.";
  answerHeading.focus();
}

function showTechnicalError(message) {
  requestStatusSection.hidden = true;
  resultSection.hidden = true;
  errorMessage.textContent = message;
  errorSection.hidden = false;
  liveStatus.textContent = "";
  errorHeading.focus();
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

async function submitQuestion(rawQuestion) {
  if (isLoading) {
    return;
  }

  const question = rawQuestion.trim();
  if (!question) {
    showValidation("Enter a question before submitting.");
    return;
  }

  clearValidation();
  lastQuestion = question;
  questionInput.value = question;
  setLoading(true, question);

  try {
    const payload = await requestAnswer(question);
    setLoading(false);
    showResult(question, payload);
  } catch (error) {
    setLoading(false);
    const message = error instanceof DisplayError ? error.message : GENERIC_ERROR;
    showTechnicalError(message);
  }
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

retryButton.addEventListener("click", () => {
  submitQuestion(lastQuestion);
});

checkHealth();

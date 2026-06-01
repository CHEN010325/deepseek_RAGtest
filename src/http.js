export class HttpError extends Error {
  constructor(message, { status = 0, body = "", url = "" } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.body = body;
    this.url = url;
  }
}

export async function requestJson(url, { method = "GET", headers = {}, body = null, timeout = 60 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout * 1000);
  let response;
  try {
    response = await fetch(url, { method, headers, body, signal: controller.signal });
  } catch (error) {
    throw new HttpError(`${method} ${url} failed: ${error.message}`, { url });
  } finally {
    clearTimeout(timer);
  }
  const text = await response.text();
  const preview = text.slice(0, 1000).replace(/\n/gu, " ");
  if (response.status >= 400) {
    throw new HttpError(`${method} ${url} returned HTTP ${response.status}: ${preview}`, {
      status: response.status,
      body: preview,
      url
    });
  }
  try {
    return text ? JSON.parse(text) : {};
  } catch (error) {
    throw new HttpError(`${method} ${url} did not return JSON: ${preview}`, {
      status: response.status,
      body: preview,
      url
    });
  }
}

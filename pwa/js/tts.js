// Spoken output with the Web Speech synthesis API (design 3.1).
// Failure tolerant: if speechSynthesis is missing or blocked, speak() does nothing.

let enabled = true;
let voice = null;

function synth() {
  return typeof globalThis.speechSynthesis !== 'undefined' ? globalThis.speechSynthesis : null;
}

function pickVoice() {
  const s = synth();
  if (!s || voice) return voice;
  try {
    const vs = s.getVoices() || [];
    voice = vs.find((v) => /^en(-|_)/i.test(v.lang) && v.localService) || vs.find((v) => /^en/i.test(v.lang)) || null;
  } catch { voice = null; }
  return voice;
}

export function setEnabled(on) { enabled = !!on; if (!on) cancel(); }
export function isEnabled() { return enabled; }

/**
 * Speak `text`. `interrupt` (default true) drops anything still queued, so the
 * newest state is what the user hears. Urgent phrases ("Stopped") are spoken a
 * little faster.
 */
export function speak(text, { interrupt = true, urgent = false } = {}) {
  const s = synth();
  if (!enabled || !s || !text) return false;
  try {
    if (interrupt) s.cancel();
    const u = new SpeechSynthesisUtterance(String(text));
    const v = pickVoice();
    if (v) u.voice = v;
    u.rate = urgent ? 1.15 : 1.0;
    u.volume = 1;
    s.speak(u);
    return true;
  } catch {
    return false;
  }
}

export function cancel() {
  try { synth()?.cancel(); } catch { /* ignore */ }
}

"use strict";
/*
 * Voice conversation mode: an open microphone, local endpointing, spoken
 * replies, and barge-in.
 *
 * This is the conversational mode. The page also keeps a simpler
 * browser-native mode (one-shot dictation via SpeechRecognition, spoken
 * replies via speechSynthesis) which needs no credentials at all — see
 * index.html. The two are deliberately separate: one is a zero-setup
 * convenience, this one is an actual conversation.
 *
 * WHY THE BROWSER DOES THE LISTENING
 * The alternative design streams raw audio to the server over a
 * WebSocket and detects turns there. Endpointing locally instead means
 * (a) barge-in costs zero network round-trips, because the microphone
 * and the speaker are both here, (b) nothing is uploaded during silence,
 * and (c) the server needs no numeric stack for frame energy. What it
 * gives up is server-side turn detection and any server-side view of the
 * waveform. At this size that is the right trade; see app/voice_api.py.
 *
 * THE BARGE-IN SEQUENCE, in the order it has to happen:
 *   1. Stop playing, now. Every scheduled buffer source is stopped and
 *      the queue is dropped. Local, so it is immediate.
 *   2. Stop synthesizing. Any sentence still in flight is abandoned and
 *      never played, so audio cannot resume behind the interruption.
 *   3. Tell the server what was actually heard, so the transcript
 *      matches the conversation. This is the step that is easy to skip
 *      and the one that keeps follow-up questions coherent — see
 *      app/conversation.py's truncate_last_reply.
 *
 * ECHO CANCELLATION IS LOAD-BEARING. With an open microphone and a
 * speaker, the agent's own voice reaches the microphone and would trip
 * the barge-in detector continuously. getUserMedia's echoCancellation
 * constraint is what makes this usable on laptop speakers rather than
 * headphones-only. It is not a nicety, and the barge-in threshold is
 * raised while speaking as a second line of defense.
 */

// ── Tuning ─────────────────────────────────────────────────────────────
// Every one of these is a real trade-off, not a magic number.
const VOICE_CONFIG = {
  // 16 kHz mono: what speech models expect. Sending the browser's native
  // 44.1/48 kHz would roughly triple the upload for no accuracy gain at
  // speech bandwidth.
  sampleRate: 16000,

  // Energy gate in dBFS. Lower catches quieter speakers and more room
  // noise; higher does the reverse. -45 is deliberately permissive,
  // because failing to hear someone is a worse failure than a false
  // start that a short minimum duration then rejects.
  speechThresholdDbfs: -45,

  // Sustained speech before we believe a turn started. Debounces a
  // cough, a keyboard click, a chair.
  speechMinMs: 200,

  // Trailing silence before we call the turn finished. The single most
  // consequential number here: too low and it cuts people off mid
  // sentence, too high and every exchange feels laggy. 800ms sits just
  // past a normal mid-sentence pause.
  silenceMs: 800,

  // Sustained speech required WHILE THE AGENT IS TALKING before it counts
  // as an interruption rather than echo or a backchannel. Higher than
  // speechMinMs on purpose: the cost of a false barge-in (cutting the
  // agent off mid-answer) is higher than the cost of a slightly late one.
  bargeInMinMs: 350,

  // Extra dB required to trigger barge-in, on top of the normal gate.
  // Echo cancellation removes most of the agent's own voice; this covers
  // what leaks through on laptop speakers at volume.
  bargeInThresholdBoostDb: 6,

  // Audio kept from BEFORE the gate opened, prepended to every captured
  // utterance. Without it, every transcript loses its first syllable —
  // by definition the gate only opens after speech has already started.
  //
  // MUST EXCEED bargeInMinMs, not just speechMinMs. Found live: at 300ms
  // this covered a normal turn (speechMinMs 200, so 100ms of headroom)
  // but NOT a barge-in, which only fires after bargeInMinMs of sustained
  // speech — 350ms of audio had already gone by, and only the last 300
  // were still buffered. The opening was discarded before _bargeIn()
  // could copy it, so interrupting the agent reliably ate the start of
  // the sentence while ordinary turns were fine. That asymmetry is the
  // signature of this bug.
  //
  // The real margin needed is larger than 350-300: during playback the
  // gate also requires bargeInThresholdBoostDb on top, and any frame
  // below it resets aboveGateMs to zero, so a quiet onset consonant can
  // push actual elapsed time well past bargeInMinMs. 500ms carries that
  // slack. Cost is ~6KB of extra 16kHz mono audio per utterance.
  preRollMs: 500,

  // A hard stop, so a stuck-open gate (a fan, a television) cannot
  // upload without bound.
  maxUtteranceMs: 30000,

  // Below this, a "turn" is a door closing, not a question. Skipped
  // without a round-trip to the transcriber.
  minUtteranceMs: 350,

  // Sentence batching for synthesis. Short sentences are merged so we do
  // not spend a network round-trip on "Yes."; long ones are the natural
  // unit and are left alone. The cap keeps time-to-first-audio low even
  // when the model writes a wall of text.
  minSpeakChars: 70,
  maxSpeakChars: 320,

  // How many synthesis requests to keep in flight ahead of playback.
  // Measured against the real endpoint, synthesis takes about two
  // seconds almost regardless of length — it is dominated by a fixed
  // per-request cost, not by how much text you send. Requesting chunks
  // strictly one at a time therefore pays that two seconds between every
  // sentence, and the reply comes out in audible steps. Overlapping
  // requests hides all but the first.
  //
  // Bounded rather than "fire them all at once" because of barge-in:
  // anything synthesized past the point where the user interrupts is
  // spend on audio nobody will ever hear. Three is the window where the
  // gaps close and the waste stays small.
  synthesisLookahead: 3,

  // Buffer size for the capture node. 2048 frames at 48 kHz is ~43ms —
  // fine-grained enough for the ms counters above to be meaningful.
  captureBufferSize: 2048,
};

/* Split a reply into speakable chunks.
 *
 * Splits only at sentence punctuation FOLLOWED BY WHITESPACE, which is
 * what keeps "0.71" and "gpt-4o-mini" intact — there is no space after
 * the dot in a decimal, so no boundary is found there. Fragments shorter
 * than minSpeakChars are merged forward ("U.S." is not a sentence), and
 * anything still over maxSpeakChars is broken at a clause boundary so
 * the first audio starts quickly.
 *
 * Exported for tests: tests/test_voice_client.py runs this file under
 * node, because the decimal case is exactly the kind of thing that
 * silently degrades speech and never throws.
 */
function splitSentences(text) {
  const normalized = String(text).replace(/\s+/g, ' ').trim();
  if (!normalized) return [];

  const raw = normalized.split(/(?<=[.!?])(?=\s)/).map(s => s.trim()).filter(Boolean);

  const merged = [];
  for (const piece of raw) {
    const last = merged[merged.length - 1];
    if (last !== undefined && last.length < VOICE_CONFIG.minSpeakChars) {
      merged[merged.length - 1] = `${last} ${piece}`;
    } else {
      merged.push(piece);
    }
  }

  const chunks = [];
  for (const piece of merged) {
    if (piece.length <= VOICE_CONFIG.maxSpeakChars) {
      chunks.push(piece);
      continue;
    }
    // Too long for one request: break at commas/semicolons/colons, which
    // are where a reader would breathe anyway.
    let remaining = piece;
    while (remaining.length > VOICE_CONFIG.maxSpeakChars) {
      const window = remaining.slice(0, VOICE_CONFIG.maxSpeakChars);
      const cut = Math.max(window.lastIndexOf(', '), window.lastIndexOf('; '), window.lastIndexOf(': '));
      const at = cut > VOICE_CONFIG.minSpeakChars ? cut + 1 : VOICE_CONFIG.maxSpeakChars;
      chunks.push(remaining.slice(0, at).trim());
      remaining = remaining.slice(at).trim();
    }
    if (remaining) chunks.push(remaining);
  }
  return chunks;
}

function dbfsOfFrame(float32) {
  if (!float32.length) return -120;
  let sumSquares = 0;
  for (let i = 0; i < float32.length; i++) sumSquares += float32[i] * float32[i];
  const rms = Math.sqrt(sumSquares / float32.length);
  return rms <= 1e-9 ? -120 : 20 * Math.log10(rms);
}

// Linear-interpolation resample to 16 kHz. Not a polyphase filter: for
// speech, into a transcriber that is itself robust to far worse, the
// difference is inaudible and the honest version is four lines.
function downsample(float32, fromRate, toRate) {
  if (fromRate === toRate) return Float32Array.from(float32);
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(float32.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const src = i * ratio;
    const lo = Math.floor(src);
    const hi = Math.min(lo + 1, float32.length - 1);
    const frac = src - lo;
    out[i] = float32[lo] * (1 - frac) + float32[hi] * frac;
  }
  return out;
}

function floatToPcm16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32[i]));
    out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return out;
}

function encodeWav(pcm16, sampleRate) {
  const buffer = new ArrayBuffer(44 + pcm16.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, 'RIFF');
  view.setUint32(4, 36 + pcm16.length * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, 'data');
  view.setUint32(40, pcm16.length * 2, true);
  for (let i = 0; i < pcm16.length; i++) view.setInt16(44 + i * 2, pcm16[i], true);
  return new Blob([buffer], { type: 'audio/wav' });
}

/* The conversation controller.
 *
 * Owns the microphone, the state machine, and playback. Knows nothing
 * about the page: everything it wants to say comes back through the
 * handlers passed to the constructor, so the DOM lives in index.html and
 * this file stays testable and readable.
 *
 * States: idle -> listening -> capturing -> transcribing -> thinking
 *         -> speaking -> listening
 */
class VoiceConversation {
  constructor(handlers) {
    this.on = handlers; // { onState, onLevel, onTranscript, onError, onSpokenPrefix, needsReply }
    this.state = 'idle';

    // Incremented by start() and by stop(). Every await in this class
    // captures it first and bails if it moved, because "the user left
    // voice mode" can happen at any of them and the work in flight has to
    // notice. Checking `state === 'idle'` instead is not enough: the
    // paths out of an await call setState themselves, so the first one to
    // run overwrites the idle that stop() just set and every later check
    // sees a live state. That is not hypothetical — ending voice mode
    // during transcription used to inject the transcript into the chat,
    // run a real billed agent turn, and surface a null-deref to the user.
    this.sessionId = 0;

    this.stream = null;
    this.captureCtx = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.sinkNode = null;
    this.playbackCtx = null;

    this.captured = [];        // Float32Array frames of the current utterance
    this.capturedSamples = 0;
    this.preRoll = [];         // rolling pre-gate window
    this.preRollSamples = 0;
    this.aboveGateMs = 0;
    this.belowGateMs = 0;

    this.playingSources = [];
    this.spokenChunks = [];
    this.speechEpoch = 0;      // bumped on every interrupt, so stale async work exits
  }

  setState(state) {
    this.state = state;
    this.on.onState?.(state);
  }

  async start() {
    const session = ++this.sessionId;

    // getUserMedia must be called from a user gesture, and so must
    // AudioContext creation — both happen on the button click that calls
    // this. echoCancellation is what makes an open mic viable next to a
    // speaker; see the file header.
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    // The permission prompt can sit open for as long as the user likes,
    // and "End voice" is reachable the whole time. Without this check the
    // microphone opens after the user has already left, with nothing
    // holding a reference to close it — a recording indicator that stays
    // lit and cannot be turned off.
    if (session !== this.sessionId) {
      stream.getTracks().forEach(t => t.stop());
      return;
    }
    this.stream = stream;

    // A device disappearing (unplugged USB mic, revoked permission) is
    // otherwise invisible: the panel keeps saying "Listening" at a dead
    // input forever.
    stream.getTracks().forEach(track => {
      track.addEventListener('ended', () => {
        if (session !== this.sessionId) return;
        this.on.onError?.('The microphone was disconnected.');
        this.stop();
      });
    });

    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.captureCtx = new Ctx();
    this.playbackCtx = new Ctx();
    await this.playbackCtx.resume();

    this.sourceNode = this.captureCtx.createMediaStreamSource(this.stream);

    // ScriptProcessorNode is deprecated in favour of AudioWorklet. Chosen
    // anyway: an AudioWorklet needs a separate module file and a message
    // port to move a 2048-frame buffer off the main thread, and that hop
    // is nowhere near the bottleneck — a network round-trip to a
    // transcription API is three orders of magnitude larger. It is
    // universally supported, and replacing it is contained to this one
    // method.
    this.processorNode = this.captureCtx.createScriptProcessor(VOICE_CONFIG.captureBufferSize, 1, 1);
    this.processorNode.onaudioprocess = (event) => this._onAudioFrame(event.inputBuffer.getChannelData(0));

    // onaudioprocess only fires while the node is connected to a graph
    // that reaches the destination. Routing through a muted gain node
    // gets the callback without piping the microphone back to the
    // speakers, which would be an unpleasant surprise.
    this.sinkNode = this.captureCtx.createGain();
    this.sinkNode.gain.value = 0;
    this.sourceNode.connect(this.processorNode);
    this.processorNode.connect(this.sinkNode);
    this.sinkNode.connect(this.captureCtx.destination);

    this._resetUtterance();
    this.setState('listening');
  }

  stop() {
    this.sessionId++;
    this.speechEpoch++;
    this._stopPlayback();
    this.processorNode?.disconnect();
    this.sourceNode?.disconnect();
    this.sinkNode?.disconnect();
    if (this.processorNode) this.processorNode.onaudioprocess = null;
    this.stream?.getTracks().forEach(t => t.stop());
    this.captureCtx?.close().catch(() => {});
    this.playbackCtx?.close().catch(() => {});
    this.stream = null;
    this.captureCtx = null;
    this.playbackCtx = null;
    this.processorNode = null;
    this.sourceNode = null;
    this.sinkNode = null;
    this._resetUtterance();
    this.setState('idle');
  }

  _resetUtterance() {
    this.captured = [];
    this.capturedSamples = 0;
    this.preRoll = [];
    this.preRollSamples = 0;
    this.aboveGateMs = 0;
    this.belowGateMs = 0;
  }

  _onAudioFrame(input) {
    const frame = downsample(input, this.captureCtx.sampleRate, VOICE_CONFIG.sampleRate);
    const frameMs = (frame.length / VOICE_CONFIG.sampleRate) * 1000;
    const dbfs = dbfsOfFrame(frame);
    this.on.onLevel?.(dbfs);

    // While the agent is speaking, the bar for "that was speech" is
    // higher — see bargeInThresholdBoostDb.
    const gate = this.state === 'speaking'
      ? VOICE_CONFIG.speechThresholdDbfs + VOICE_CONFIG.bargeInThresholdBoostDb
      : VOICE_CONFIG.speechThresholdDbfs;
    const above = dbfs >= gate;

    if (above) {
      this.aboveGateMs += frameMs;
      this.belowGateMs = 0;
    } else {
      this.belowGateMs += frameMs;
      this.aboveGateMs = 0;
    }

    // The pre-roll window runs in every state, including while the agent
    // is talking — that is what makes a barge-in's own first syllable
    // survive into the transcript.
    this.preRoll.push(frame);
    this.preRollSamples += frame.length;
    const preRollCap = (VOICE_CONFIG.preRollMs / 1000) * VOICE_CONFIG.sampleRate;
    while (this.preRollSamples > preRollCap && this.preRoll.length > 1) {
      this.preRollSamples -= this.preRoll.shift().length;
    }

    if (this.state === 'listening' && above && this.aboveGateMs >= VOICE_CONFIG.speechMinMs) {
      this.captured = this.preRoll.slice();
      this.capturedSamples = this.preRollSamples;
      this.setState('capturing');
      return;
    }

    if (this.state === 'capturing') {
      this.captured.push(frame);
      this.capturedSamples += frame.length;
      const capturedMs = (this.capturedSamples / VOICE_CONFIG.sampleRate) * 1000;
      if (this.belowGateMs >= VOICE_CONFIG.silenceMs || capturedMs >= VOICE_CONFIG.maxUtteranceMs) {
        this._finishUtterance();
      }
      return;
    }

    if (this.state === 'speaking' && above && this.aboveGateMs >= VOICE_CONFIG.bargeInMinMs) {
      this._bargeIn();
    }
  }

  _bargeIn() {
    // 1. Stop playing, now. 2. Invalidate anything still being fetched.
    this.speechEpoch++;
    this._stopPlayback();
    // 3. Tell the server what was actually heard. Fire and forget: the
    // user is already mid-sentence and must not wait on a round-trip.
    const spoken = this.spokenChunks.join(' ');
    this.on.onSpokenPrefix?.(spoken);
    fetch('/voice/interrupt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spoken_prefix: spoken }),
    }).catch(() => {});

    // Their interruption is the start of the next turn, and the pre-roll
    // already holds its opening syllables.
    this.captured = this.preRoll.slice();
    this.capturedSamples = this.preRollSamples;
    this.setState('capturing');
  }

  async _finishUtterance() {
    const frames = this.captured;
    const totalSamples = this.capturedSamples;
    this._resetUtterance();

    const durationMs = (totalSamples / VOICE_CONFIG.sampleRate) * 1000;
    if (durationMs < VOICE_CONFIG.minUtteranceMs) {
      this.setState('listening'); // a door, not a question
      return;
    }

    const session = this.sessionId;
    this.setState('transcribing');
    const merged = new Float32Array(totalSamples);
    let offset = 0;
    for (const frame of frames) {
      merged.set(frame, offset);
      offset += frame.length;
    }
    const wav = encodeWav(floatToPcm16(merged), VOICE_CONFIG.sampleRate);

    let transcript = '';
    try {
      const res = await fetch('/voice/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: wav,
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      transcript = ((await res.json()).text || '').trim();
    } catch (err) {
      if (session !== this.sessionId) return; // left voice mode; not the user's problem
      this.on.onError?.('Could not transcribe that — the speech service did not respond.');
      this.setState('listening');
      return;
    }
    if (session !== this.sessionId) return; // left voice mode while transcribing

    // Transcribers emit something for almost any sound. A turn with no
    // letters in it was noise, and sending it to the agent would produce
    // a confident answer to a question nobody asked.
    if (!/[a-z]/i.test(transcript)) {
      this.setState('listening');
      return;
    }

    this.on.onTranscript?.(transcript);
    this.setState('thinking');
    let reply = null;
    try {
      reply = await this.on.needsReply?.(transcript);
    } catch (err) {
      if (session !== this.sessionId) return;
      this.on.onError?.('The agent could not answer that one.');
    }
    if (session !== this.sessionId) return; // left voice mode mid-turn
    if (reply) await this.speak(reply);
    else this.setState('listening');
  }

  _stopPlayback() {
    for (const source of this.playingSources) {
      try { source.stop(0); } catch (e) { /* already ended */ }
    }
    this.playingSources = [];
  }

  /* Speak a reply, one chunk at a time, fetching the next while the
   * current one plays. Sequential rather than all-at-once because the
   * point of chunking is that audio starts before synthesis finishes;
   * firing every request in parallel would spend the same wall-clock time
   * and lose the ability to abandon the rest on an interruption. */
  async speak(text) {
    const spokenText = typeof stripMarkdown === 'function' ? stripMarkdown(text) : text;
    const chunks = splitSentences(spokenText);
    if (!chunks.length) {
      this.setState('listening');
      return;
    }

    const epoch = ++this.speechEpoch;
    const session = this.sessionId;
    this.spokenChunks = [];
    this.setState('speaking');

    const fetchAudio = async (chunk) => {
      const res = await fetch('/voice/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: chunk }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      return this.playbackCtx.decodeAudioData(await res.arrayBuffer());
    };

    const inFlight = [];
    let nextToFetch = 0;
    const topUp = () => {
      while (inFlight.length < VOICE_CONFIG.synthesisLookahead && nextToFetch < chunks.length) {
        const request = fetchAudio(chunks[nextToFetch++]);
        // A prefetch can reject while we are still playing an earlier
        // chunk and not yet awaiting it. Attaching a handler now marks it
        // handled, so a failed prefetch does not surface as an unhandled
        // rejection; the error is still raised at the await below, where
        // there is somewhere sensible to report it.
        request.catch(() => {});
        inFlight.push(request);
      }
    };

    try {
      topUp();
      for (let i = 0; i < chunks.length; i++) {
        const buffer = await inFlight.shift();
        if (epoch !== this.speechEpoch || session !== this.sessionId) return; // interrupted or left
        topUp();
        await this._playBuffer(buffer, epoch);
        if (epoch !== this.speechEpoch) return; // interrupted while playing
        // Counted as spoken only once it has finished playing, so the
        // prefix sent on barge-in is never optimistic.
        this.spokenChunks.push(chunks[i]);
      }
    } catch (err) {
      if (epoch === this.speechEpoch && session === this.sessionId) {
        this.on.onError?.('Could not play the spoken reply — it is still on screen above.');
      }
    }

    if (epoch === this.speechEpoch && session === this.sessionId) this.setState('listening');
  }

  _playBuffer(buffer, epoch) {
    return new Promise((resolve) => {
      const source = this.playbackCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(this.playbackCtx.destination);
      source.onended = () => {
        this.playingSources = this.playingSources.filter(s => s !== source);
        resolve();
      };
      this.playingSources.push(source);
      if (epoch !== this.speechEpoch) { resolve(); return; }
      source.start();
    });
  }
}

// Browser: globals, which is all index.html needs. Node (the test
// harness): CommonJS, so the tests exercise the shipped file rather than
// a copy that can drift away from it.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { splitSentences, dbfsOfFrame, downsample, floatToPcm16, VOICE_CONFIG };
}

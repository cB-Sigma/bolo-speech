/* tslint:disable */
/* eslint-disable */

/**
 * A vocabulary prepared for sound-alike matching: a word list (TSV, PLS
 * or JSON), a JSON array of terms, or one term per line; `options` is a
 * JSON object (`langs`) or empty.
 */
export class SoundIndex {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The corrected text with its changes, as JSON; options as JSON.
     */
    applyJson(text: string, lang_tag: string, options: string): string;
    /**
     * Candidates as JSON; options as JSON.
     */
    findJson(text: string, lang_tag: string, options: string): string;
    /**
     * Build the index.
     */
    constructor(vocabulary: string, options: string);
    /**
     * Number of terms.
     */
    readonly length: number;
}

/**
 * Convert a word list (TSV, PLS or JSON text) to `tsv`, `json` or `pls`
 * (`lang` picks the PLS language; empty for entries of any language).
 */
export function convertWordlist(text: string, to: string, lang_tag: string): string;

/**
 * True when the language's data has been loaded.
 */
export function has(lang_tag: string): boolean;

/**
 * True when misaki's dictionaries are loaded for the language.
 */
export function hasMisaki(lang_tag: string): boolean;

/**
 * True when the transliteration pack of the language is loaded.
 */
export function hasTranslit(lang_tag: string): boolean;

/**
 * Kitten TTS 0.8 input ids for a phoneme string.
 */
export function kittenIds(phonemes: string): Int32Array;

/**
 * Kokoro input ids for a token string.
 */
export function kokoroIds(tokens: string): Int32Array;

/**
 * Load the data image of a language.
 */
export function load(lang_tag: string, bytes: Uint8Array | number[]): void;

/**
 * Load a compiled Kokoro English data file (`kokoro-en-us.ucd` /
 * `kokoro-en-gb.ucd`). Returns the language tag it is for.
 */
export function loadKokoro(bytes: Uint8Array | number[]): string;

/**
 * Load misaki's dictionaries (the text of its gold and silver JSON files)
 * for the Kokoro English front end.
 */
export function loadMisaki(lang_tag: string, gold: string, silver: string): void;

/**
 * Load any `.ucd` pack (for example an optional `wikipron-*` pack).
 * Returns the pack's name.
 */
export function loadPack(bytes: Uint8Array | number[]): string;

/**
 * Matcha-TTS input ids for a phoneme string.
 */
export function matchaIds(phonemes: string): Int32Array;

/**
 * Spoken form with spans, as JSON; options as JSON (word lists as text
 * in its `wordlists` member).
 */
export function normalizeJson(text: string, lang_tag: string, options: string): string;

/**
 * One string in the given profile.
 */
export function render(text: string, lang_tag: string, profile_name: string, normalize: string, improved: boolean, hinglish: boolean, punctuation: boolean, flags: boolean): string;

/**
 * `render` with word lists (TSV, PLS or JSON texts) applied.
 */
export function renderWordlists(text: string, lang_tag: string, profile_name: string, normalize: string, improved: boolean, hinglish: boolean, punctuation: boolean, flags: boolean, lists: string[]): string;

/**
 * Respell IPA for a text voice: JSON `{"text","readback","distance"}`, or
 * an empty string when the language has no writer.
 */
export function respell(ipa: string, lang_tag: string, source: string, style: string): string;

/**
 * Piper-shaped sentence strings.
 */
export function sentences(text: string, lang_tag: string, normalize: string, improved: boolean, hinglish: boolean): string[];

/**
 * Romanised text into the script of the language, with spans, as JSON;
 * options as JSON (`english`, `alternatives`, `context`).
 */
export function transliterateJson(text: string, lang_tag: string, options: string): string;

/**
 * Library version.
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_soundindex_free: (a: number, b: number) => void;
    readonly convertWordlist: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly has: (a: number, b: number) => number;
    readonly hasMisaki: (a: number, b: number) => number;
    readonly hasTranslit: (a: number, b: number) => number;
    readonly kittenIds: (a: number, b: number, c: number) => void;
    readonly kokoroIds: (a: number, b: number, c: number) => void;
    readonly load: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly loadKokoro: (a: number, b: number, c: number) => void;
    readonly loadMisaki: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly loadPack: (a: number, b: number, c: number) => void;
    readonly matchaIds: (a: number, b: number, c: number) => void;
    readonly normalizeJson: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly render: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => void;
    readonly renderWordlists: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => void;
    readonly respell: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly sentences: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly soundindex_applyJson: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly soundindex_findJson: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly soundindex_length: (a: number) => number;
    readonly soundindex_new: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly transliterateJson: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly version: (a: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export4: (a: number) => void;
    readonly __wbindgen_export5: () => number;
    readonly __wbindgen_export6: (a: number, b: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

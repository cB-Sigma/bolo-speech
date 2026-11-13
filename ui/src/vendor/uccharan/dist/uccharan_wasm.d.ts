/* tslint:disable */
/* eslint-disable */

/**
 * True when the language's data has been loaded.
 */
export function has(lang_tag: string): boolean;

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
 * Matcha-TTS input ids for a phoneme string.
 */
export function matchaIds(phonemes: string): Int32Array;

/**
 * One string in the given profile.
 */
export function render(text: string, lang_tag: string, profile_name: string, normalize: string, improved: boolean, hinglish: boolean, punctuation: boolean, flags: boolean): string;

/**
 * Piper-shaped sentence strings.
 */
export function sentences(text: string, lang_tag: string, normalize: string, improved: boolean, hinglish: boolean): string[];

/**
 * Library version.
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly has: (a: number, b: number) => number;
    readonly kittenIds: (a: number, b: number, c: number) => void;
    readonly kokoroIds: (a: number, b: number, c: number) => void;
    readonly load: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly matchaIds: (a: number, b: number, c: number) => void;
    readonly render: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => void;
    readonly sentences: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly version: (a: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export4: (a: number) => void;
    readonly __wbindgen_export5: (a: number, b: number) => void;
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

// Forwards microphone audio to the main thread in ~20 ms blocks.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.size = Math.round(sampleRate * 0.02)
    this.buffer = new Float32Array(this.size)
    this.fill = 0
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (!channel) return true
    let offset = 0
    while (offset < channel.length) {
      const take = Math.min(this.size - this.fill, channel.length - offset)
      this.buffer.set(channel.subarray(offset, offset + take), this.fill)
      this.fill += take
      offset += take
      if (this.fill === this.size) {
        this.port.postMessage(this.buffer, [this.buffer.buffer])
        this.buffer = new Float32Array(this.size)
        this.fill = 0
      }
    }
    return true
  }
}
registerProcessor('bolo-capture', CaptureProcessor)

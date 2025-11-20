import { pipeline } from '@huggingface/transformers'

let extractor = null

export async function loadModel(modelName, setStatus) {
  setStatus('Downloading and initializing model...')

  // 1. Create the text-generation pipeline with the small, optimized model
  extractor = await pipeline('text-generation', modelName, {
    // Use 4-bit quantization for minimal file size and faster load
    quantized: true,
    // Attempt to use WebGPU for max speed, falls back to WASM/CPU if unavailable
    device: 'webgpu'
  })
}

export async function extract(systemPrompt, input, onSuccess, onStatus, onError) {
  if (!extractor) {
    onError({ message: 'Model is not initialized.' })
    return
  }

  // Combine system instruction and user input into a single prompt
  const fullPrompt = `${systemPrompt}\n\nUSER INPUT:\n${input}`

  onStatus('Analyzing conversation... (Processing in browser)');

  // 2. Run the model inference
  await extractor(fullPrompt, {
    // Parameters to force structured output (crucial for extraction)
    max_new_tokens: 500,
    temperature: 0.1, // Keep it low for deterministic extraction
    // Force the output to start with JSON syntax
    prefix_tokens: extractor.tokenizer('```json\n', { add_special_tokens: false }).input_ids,
    // A common technique for cleaning up LLM JSON output
    stop: ['```']
  })
    .then(
      (result) => {
        // 3. Clean and parse the raw LLM output
        const rawText = result[0].generated_text.trim()
        onSuccess(rawText);
      }
    )
    .catch(onError);
}

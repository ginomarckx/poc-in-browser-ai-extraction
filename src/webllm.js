import * as webllm from '@mlc-ai/web-llm';

/*************** WebLLM logic ***************/
// Create engine instance
const engine = new webllm.MLCEngine();

export async function loadModel(modelName, setStatus) {
    engine.setInitProgressCallback((report) => {
        console.log("initialize", report.progress);
        setStatus(report.text);
    });

    const config = {
        temperature: 1.0,
        top_p: 1,
    };
    await engine.reload(modelName, config);
}

async function streamingGenerating(messages, onUpdate, onFinish, onError) {
    try {
        const completion = await engine.chat.completions.create({
            stream: true,
            messages,
        });
        // Not sure if we need this, this is for intermediate results, might turn off stream above
        let curMessage = "";
        for await (const chunk of completion) {
            const curDelta = chunk.choices[0].delta.content;
            if (curDelta) {
                curMessage += curDelta;
            }
            console.log("curMessage", curMessage);
            onUpdate(curMessage);
        }
        const finalMessage = await engine.getMessage();
        console.log("finalMessage", finalMessage);
        onFinish(finalMessage);
    } catch (err) {
        onError(err);
    }
}

/*************** UI logic ***************/
export async function extract(systemPrompt, input, onSuccess, onStatus, onError) {
    const prompt = [
        {
            content: systemPrompt,
            role: "system",
        },
        {
            content: input,
            role: "user",
        },
    ];

    streamingGenerating(
        prompt,
        onStatus,
        onSuccess,
        onError,
    );
}
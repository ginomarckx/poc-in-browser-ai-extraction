import * as webllm from '@mlc-ai/web-llm';

let currentModule = null;

const statusElement = document.getElementById('status')
const technologySelect = document.getElementById('technologySelect');
const modelSelect = document.getElementById('modelSelect');
const loadModelButton = document.getElementById('loadModelButton');

const transformersModels = [
    'Xenova/OpenELM-270M-Instruct',
    'Xenova/Phi-3-mini-4k-instruct'
];

const webllmModels = webllm.prebuiltAppConfig.model_list.map(
    (m) => m.model_id,
);

function setStatus(status, style) {
    statusElement.textContent = status;
    statusElement.className = style;
}

function selectTechnology() {
    const models = technologySelect.value === 'transformers' ? transformersModels : webllmModels;

    modelSelect.innerHTML = '<option disabled selected value>Select a model</option>';
    models.forEach((m) => {
        const option = document.createElement("option");
        option.value = m;
        option.textContent = m;
        modelSelect.appendChild(option);
    });

    loadModelButton.disabled = true;
}

function selectModel() {
    loadModelButton.disabled = false;
}

function loadModel(model) {
    setStatus('Loading model... (This may take several minutes on first run)', 'loading');
    currentModule.loadModel(
        model,
        (status) => setStatus(status, 'loading')
    )
        .then(
            () => {
                setStatus('Model Ready (Private, On-Device Inference)', 'ready');
                loadModelButton.disabled = false;
                extractButton.disabled = false;
            }
        )
        .catch(
            (error) => {
                setStatus(`Error loading model: ${error.message}. Please check console.`, 'error');
                console.error('Model Loading Error:', error)
            }
        )
}

function loadModule() {
    currentModule = null;
    document.getElementById('technology-module')?.remove();
    const currentModel = document.getElementById('current-model');

    currentModel.textContent = ''
    loadModelButton.disabled = true;
    setStatus('Loading modulel...', 'loading');

    const technology = technologySelect.value;
    const model = modelSelect.value;

    const moduleSrc = technology === 'transformers' ? './transformers.js' : './webllm.js';

    import(moduleSrc)
        .then((module) => {
            currentModule = module;
            currentModel.textContent = `${technology} - ${model}`;
            loadModel(model);
        })
        .catch((error) => {
            setStatus(`Error loading module: ${error.message}. Please check console.`, 'error');
            console.error('Module Loading Error:', error);
        })
}

function extract() {
    const systemPrompt = document.getElementById('systemPrompt').value.trim();
    const input = document.getElementById('conversationInput').value.trim();
    if (!input) {
        setStatus('Please enter a conversation transcript.', 'error');
        return;
    }

    setStatus('Running extraction...', 'loading');
    currentModule.extract(
        systemPrompt,
        input,
        (output) => {
            document.getElementById('output').textContent = output;
            setStatus('Extraction Complete!', 'ready');
        },
        (status) => {
            setStatus(status, 'loading');
        },
        (error) => {
            setStatus(`Error running extraction: ${error.message}. Please check console.`, 'error');
            console.error('Extraction Error:', error);
        }
    );
}

technologySelect.addEventListener('change', selectTechnology);
modelSelect.addEventListener('change', selectModel);
loadModelButton.addEventListener('click', loadModule);

extractButton.addEventListener('click', extract)

setStatus("Model not loaded", 'error');
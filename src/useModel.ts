import { useEffect, useState } from 'react';
import { MobileModel, Module, torch } from 'react-native-pytorch-core';

/**
 * The useModel hook loads a PyTorch model for lite interpreter from a given
 * url and then loads it as a lite interpreter model into memory.
 *
 * @param url A public url to a PyTorch model for lite interpreter.
 * @returns An isReady value indicating when the model is ready (true) and the
 * model when loaded.
 */
export default function useModel(url: string) {
    const [isReady, setIsReady] = useState(false);
    const [model, setModel] = useState<Module | null>(null);
    useEffect(() => {
        setIsReady(false);
        async function loadModel() {
            console.log('Downloading model from', url);
            const filePath = await MobileModel.download(url);
            console.log('Model downloaded to', filePath);
            const model = await torch.jit._loadForMobile(filePath);
            console.log('Model loaded for lite interpreter');
            setModel(model);
            setIsReady(true);
        }
        loadModel();
    }, [setIsReady, setModel, url]);

    return {
        isReady,
        model,
    }
}

import React from 'react';
import { Phase } from "./Walkthrough";
import { commentary, embed, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Vec3 } from '@/src/utils/vector';

let minGptLink = 'https://github.com/karpathy/minGPT';
let pytorchLink = 'https://pytorch.org/';
let andrejLink = 'https://karpathy.ai/';
let zeroToHeroLink = 'https://karpathy.ai/zero-to-hero.html';

export function walkthrough01_Prelim(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;

    if (wt.phase !== Phase.Intro_Prelim) {
        return;
    }

    setInitialCamera(state, new Vec3(184.744, 0.000, -636.820), new Vec3(296.000, 16.000, 13.500));

    let c0 = commentary(wt, null, 0)`
Before we delve into the algorithm's intricacies, let's take a brief step back.

This guide focuses on _inference_, not training, and as such is only a small part of the entire machine-learning process.
In our case, the model's weights have been pre-trained, and we use the inference process to generate output. This runs directly in your browser.

The model showcased here is part of the GPT (generative pre-trained transformer) family, which can be described as a "context-based token predictor".
OpenAI introduced this family in 2018, with notable members such as GPT-2, GPT-3, and GPT-3.5 Turbo, the latter being the foundation of the widely-used ChatGPT.
It might also be related to GPT-4, but specific details remain unknown.

This guide was inspired by the ${embedLink('minGPT', minGptLink)} GitHub project, a minimal GPT implementation in ${embedLink('PyTorch', pytorchLink)}
created by ${embedLink('Andrej Karpathy', andrejLink)}.
His YouTube series ${embedLink("Neural Networks: Zero to Hero", zeroToHeroLink)} and the minGPT project have been invaluable resources in the creation of this
guide. The toy model featured here is based on one found within the minGPT project.

Alright, let's get started!
`.zh`

在深入演算法的細節之前，我們先稍微退一步看看全貌。

本導覽聚焦在「推論」（_inference_），而不是訓練，因此只涵蓋整個機器學習流程中的一小部分。在這裡，模型的權重已經事先訓練好，我們用推論過程來產生輸出。這一切都直接在你的瀏覽器裡執行。

這裡展示的模型屬於 GPT（generative pre-trained transformer，生成式預訓練 transformer）家族，可以描述為一種「根據上下文預測 token 的機器」。OpenAI 在 2018 年推出這個家族，知名成員包括 GPT-2、GPT-3 與 GPT-3.5 Turbo，其中 GPT-3.5 Turbo 正是廣為使用的 ChatGPT 的基礎。它可能也與 GPT-4 有關，但具體細節目前仍不得而知。

本導覽的靈感來自 GitHub 專案 ${embedLink('minGPT', minGptLink)}，這是 ${embedLink('Andrej Karpathy', andrejLink)} 以 ${embedLink('PyTorch', pytorchLink)} 寫成的極簡 GPT 實作。他的 YouTube 系列 ${embedLink("Neural Networks: Zero to Hero", zeroToHeroLink)} 與 minGPT 專案，都是製作本導覽時非常寶貴的資源。這裡展示的迷你模型，就是以 minGPT 專案中的一個模型為基礎。

好，我們開始吧！

`;

}

export function embedLink(a: React.ReactNode, href: string) {
    return embedInline(<a className={s.externalLink} href={href} target="_blank" rel="noopener noreferrer">{a}</a>);
}

export function embedInline(a: React.ReactNode) {
    return { insertInline: a };
}


// Another similar model is BERT (bidirectional encoder representations from transformers), a "context-aware text encoder" commonly
// used for tasks like document classification and search.  Newer models like Facebook's LLaMA (large language model architecture), continue to use
// a similar transformer architecture, albeit with some minor differences.

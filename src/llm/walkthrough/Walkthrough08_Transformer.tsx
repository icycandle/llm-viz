import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));

    let c0 = commentary(wt, null, 0)`

And that's a complete transformer block!

These form the bulk of any GPT model and are repeated a number of times, with the output of one
block feeding into the next, continuing the residual pathway.

As is common in deep learning, it's hard to say exactly what each of these layers is doing, but we
have some general ideas: the earlier layers tend to focus on learning
lower-level features and patterns, while the later layers learn to recognize and understand
higher-level abstractions and relationships. In the context of natural language processing, the
lower layers might learn grammar, syntax, and simple word associations, while the higher layers
might capture more complex semantic relationships, discourse structures, and context-dependent meaning.

`.zh`


這就是一個完整的 transformer 區塊！

這些區塊構成了所有 GPT 模型的主體，會重複好幾次：一個區塊的輸出送進下一個區塊，並延續 residual pathway。

和深度學習中常見的情況一樣，我們很難確切說出每一層在做什麼，但大致有些概念：較前面的層傾向學習較低階的特徵與模式，而較後面的層則學會辨識與理解較高階的抽象概念與關係。以自然語言處理來說，較低的層可能學到文法、句法與簡單的詞語關聯，而較高的層可能捕捉到更複雜的語意關係、篇章結構，以及依賴上下文的意義。

`;

}

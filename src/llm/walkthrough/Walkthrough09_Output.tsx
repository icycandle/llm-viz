import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough09_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Output) {
        return;
    }

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

    let c0 = commentary(wt, null, 0)`

Finally, we come to the end of the model. The output of the final transformer block is passed through
a layer normalization, and then we use a linear transformation (matrix multiplication), this time without a bias.

This final transformation takes each of our column vectors from length C to length nvocab. Hence,
it's effectively producing a score for each word in the vocabulary for each of our columns. These
scores have a special name: logits.

The name "logits" comes from "log-odds," i.e., the logarithm of the odds of each token. "Log" is
used because the softmax we apply next does an exponentiation to convert to "odds" or probabilities.

To convert these scores into nice probabilities, we pass them through a softmax operation. Now, for
each column, we have a probability the model assigns to each word in the vocabulary.

In this particular model, it has effectively learned all the answers to the question of how to sort
three letters, so the probabilities are heavily weighted toward the correct answer.

When we're stepping the model through time, we use the last column's probabilities to determine the
next token to add to the sequence. For example, if we've supplied six tokens into the model, we'll
use the output probabilities of the 6th column.

This column's output is a series of probabilities, and we actually have to pick one of them to use
as the next in the sequence. We do this by "sampling from the distribution." That is, we randomly
choose a token, weighted by its probability. For example, a token with a probability of 0.9 will be
chosen 90% of the time.

There are other options here, however, such as always choosing the token with the highest probability.

We can also control the "smoothness" of the distribution by using a temperature parameter. A higher
temperature will make the distribution more uniform, and a lower temperature will make it more
concentrated on the highest probability tokens.

We do this by dividing the logits (the output of the linear transformation) by the temperature before
applying the softmax. Since the exponentiation in the softmax has a large effect on larger numbers,
making them all closer together will reduce this effect.
`.zh`


終於，我們來到模型的最末端。最後一個 transformer 區塊的輸出會先經過一次 layer normalization，接著做一次線性轉換（矩陣乘法），這次不加偏差。

這最後的轉換把每個欄向量從長度 C 變成長度 nvocab。因此，它實際上是為每一欄、針對詞彙表中的每個字各產生一個分數。這些分數有個特別的名字：logits。

「logits」這個名稱來自「log-odds」，也就是每個 token 勝算（odds）的對數。之所以用「log」，是因為接下來套用的 softmax 會做指數運算，把它轉換成「勝算」或機率。

為了把這些分數轉換成漂亮的機率，我們讓它們通過 softmax 運算。現在，對每一欄來說，模型都為詞彙表中的每個字分配了一個機率。

在這個模型中，它實際上已經學會了「如何排序三種字母」這個問題的所有答案，所以機率會大幅集中在正確答案上。

當我們讓模型一步步往前推進時，會用最後一欄的機率來決定要加到序列中的下一個 token。例如，如果我們已經輸入了六個 token，就使用第 6 欄的輸出機率。

這一欄的輸出是一串機率，而我們必須從中實際挑出一個，作為序列中的下一個 token。做法是「從分布中取樣」，也就是依照機率加權，隨機選出一個 token。例如，機率為 0.9 的 token，有 90% 的機會被選中。

不過也有其他做法，例如永遠選擇機率最高的 token。

我們也可以用溫度（temperature）參數來控制分布的「平滑程度」。溫度越高，分布越平均；溫度越低，分布越集中在機率最高的幾個 token 上。

做法是在套用 softmax 之前，先把 logits（線性轉換的輸出）除以溫度。由於 softmax 中的指數運算對較大的數字影響很大，把它們除以溫度、讓彼此更接近，就能減弱這個效果。

`;

}

import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough05_Softmax(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Softmax) {
        return;
    }

    setInitialCamera(state, new Vec3(-24.350, 0.000, -1702.195), new Vec3(283.100, 0.600, 1.556));

    let c0 = commentary(wt, null, 0)`

The softmax operation is used as part of self-attention, as seen in the previous section, and it
will also appear at the very end of the model.

Its goal is to take a vector and normalize its values so that they sum to 1.0. However, it's not as
simple as dividing by the sum. Instead, each input value is first exponentiated.

  a = exp(x_1)

This has the effect of making all values positive. Once we have a vector of our exponentiated
values, we can then divide each value by the sum of all the values. This will ensure that the sum
of the values is 1.0. Since all the exponentiated values are positive, we know that the resulting
values will be between 0.0 and 1.0, which provides a probability distribution over the original values.

That's it for softmax: simply exponentiate the values and then divide by the sum.

However, there's a slight complication. If any of the input values are quite large, then the
exponentiated values will be very large. We'll end up dividing a large number by a very large number,
and this can cause issues with floating-point arithmetic.

One useful property of the softmax operation is that if we add a constant to all the input values,
the result will be the same. So we can find the largest value in the input vector and subtract it
from all the values. This ensures that the largest value is 0.0, and the softmax remains numerically
stable.

Let's take a look at the softmax operation in the context of the self-attention layer. Our input
vector for each softmax operation is a row of the self-attention matrix (but only up to the diagonal).

Like with layer normalization, we have an intermediate step where we store some aggregation values
to keep the process efficient.

For each row, we store the max value in the row and the sum of the shifted & exponentiated values.
Then, to produce the corresponding output row, we can perform a small set of operations: subtract the
max, exponentiate, and divide by the sum.

What's with the name "softmax"? The "hard" version of this operation, called argmax, simply finds
the maximum value, sets it to 1.0, and assigns 0.0 to all other values. In contrast, the softmax
operation serves as a "softer" version of that. Due to the exponentiation involved in softmax, the
largest value is emphasized and pushed towards 1.0, while still maintaining a probability distribution
over all input values. This allows for a more nuanced representation that captures not only the most
likely option but also the relative likelihood of other options.
`.zh`


softmax 運算是 self-attention 的一部分（如上一節所見），它也會出現在模型的最末端。

它的目標是把一個向量的數值正規化，使其總和為 1.0。不過，這並不是單純除以總和那麼簡單。每個輸入值會先取指數。

  a = exp(x_1)

這麼做的效果是讓所有數值都變成正數。得到取指數後的向量之後，再把每個數值除以所有數值的總和，就能確保總和為 1.0。由於取指數後的值都是正數，結果一定介於 0.0 到 1.0 之間，於是就得到原始數值上的一個機率分布。

softmax 就是這樣：把數值取指數，再除以總和。

不過，這裡有個小麻煩。如果某些輸入值相當大，取指數後的值就會變得非常大。我們最後會拿一個大數除以一個更大的數，這可能會造成浮點數運算上的問題。

softmax 有一個好用的性質：如果把所有輸入值都加上同一個常數，結果不會改變。所以我們可以找出輸入向量中的最大值，讓所有數值都減去它。這樣能確保最大值是 0.0，softmax 在數值上也就保持穩定。

我們來看看 softmax 運算在 self-attention 層中的情況。每次 softmax 運算的輸入向量，是 self-attention 矩陣的一列（但只到對角線為止）。

和 layer normalization 一樣，這裡也有一個中間步驟，會先存下一些彙總值，讓整個流程更有效率。

對每一列，我們會存下該列的最大值，以及「平移並取指數後」各值的總和。接著，要產生對應的輸出列時，只需要做幾個簡單的運算：減去最大值、取指數、再除以總和。

為什麼叫「softmax」呢？這個運算的「硬」版本叫 argmax，它只是找出最大值、把它設為 1.0，其他值全部設為 0.0。相較之下，softmax 是它的「軟」版本。由於 softmax 中的取指數運算，最大的值會被強調並推向 1.0，同時仍保持在所有輸入值上的機率分布。這讓它能表達更細緻的資訊：不只捕捉最可能的選項，也保留其他選項的相對可能性。

`;

}

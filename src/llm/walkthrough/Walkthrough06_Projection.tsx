import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough06_Projection(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.Input_Detail_Projection) {
        return;
    }

    setInitialCamera(state, new Vec3(-73.167, 0.000, -270.725), new Vec3(293.606, 2.613, 1.366));
    let block = layout.blocks[0];
    wt.dimHighlightBlocks = [...block.heads.map(h => h.vOutBlock), block.projBias, block.projWeight, block.attnOut];

    let outBlocks = block.heads.map(h => h.vOutBlock);

    commentary(wt, null, 0)`

After the self-attention process, we have outputs from each of the heads. These outputs are the
appropriately mixed V vectors, influenced by the Q and K vectors.

To combine the ${c_blockRef('output vectors', outBlocks)} from each head, we simply stack them on top of each other. So, for time
${c_dimRef('t = 4', DimStyle.T)}, we go from 3 vectors of length ${c_dimRef('A = 16', DimStyle.A)} to 1 vector of length ${c_dimRef('C = 48', DimStyle.C)}.`.zh`


self-attention 處理完之後，每個 head 都會產生輸出。這些輸出是經過適當混合的 V 向量，混合方式受 Q 和 K 向量影響。

要合併各個 head 的 ${c_blockRef('輸出向量', outBlocks)}，我們只要把它們上下堆疊起來。所以在時間 ${c_dimRef('t = 4', DimStyle.T)}，我們從 3 個長度為 ${c_dimRef('A = 16', DimStyle.A)} 的向量，變成 1 個長度為 ${c_dimRef('C = 48', DimStyle.C)} 的向量。`;

    breakAfter();

    let t_fadeOut = afterTime(null, 1.0, 0.5);
    // let t_zoomToStack = afterTime(null, 1.0);
    let t_stack = afterTime(null, 1.0);

    breakAfter();

    commentary(wt)`

It's worth noting that in GPT, the length of the vectors within a head (${c_dimRef('A = 16', DimStyle.A)}) is equal to ${c_dimRef('C', DimStyle.C)} / num_heads.
This ensures that when we stack them back together, we get the original length, ${c_dimRef('C', DimStyle.C)}.

From here, we perform the projection to get the output of the layer. This is a simple matrix-vector
multiplication on a per-column basis, with a bias added.`.zh`


值得一提的是，在 GPT 中，head 內向量的長度（${c_dimRef('A = 16', DimStyle.A)}）等於 ${c_dimRef('C', DimStyle.C)} / head 數量。這確保了堆疊回去之後，長度恰好回到原本的 ${c_dimRef('C', DimStyle.C)}。

接著，我們做投影（projection）來得到這一層的輸出。這只是對每一欄各自做一次矩陣與向量的乘法，再加上偏差。`;

    breakAfter();

    let t_process = afterTime(null, 3.0);

    breakAfter();

    commentary(wt)`

Now we have the output of the self-attention layer. Instead of passing this output directly to the
next phase, we add it element-wise to the input embedding. This process, denoted by the green
vertical arrow, is called the _residual connection_ or _residual pathway_.
`.zh`


現在我們得到了 self-attention 層的輸出。我們不直接把這個輸出傳到下一個階段，而是把它逐元素加回輸入 embedding 上。這個過程以綠色的垂直箭頭表示，稱為 _residual connection_（殘差連接）或 _residual pathway_（殘差路徑）。

`;

    breakAfter();

    let t_zoomOut = afterTime(null, 1.0, 0.5);
    let t_processResid = afterTime(null, 3.0);

    cleanup(t_zoomOut, [t_fadeOut, t_stack]);

    breakAfter();

    commentary(wt)`

Like layer normalization, the residual pathway is important for enabling effective learning in deep
neural networks.

Now with the result of self-attention in hand, we can pass it onto the next section of the transformer:
the feed-forward network.
`.zh`


和 layer normalization 一樣，residual pathway 對於讓深度神經網路有效學習非常重要。

拿到 self-attention 的結果後，我們就可以把它送進 transformer 的下一個部分：前饋網路（feed-forward network）。

`;

    breakAfter();

    if (t_fadeOut.active) {
        for (let head of block.heads) {
            for (let blk of head.cubes) {
                if (blk !== head.vOutBlock) {
                    blk.opacity = lerpSmoothstep(1, 0, t_fadeOut.t);
                }
            }
        }
    }

    if (t_stack.active) {
        let targetZ = block.attnOut.z;
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            let head = block.heads[headIdx];
            let targetY = head.vOutBlock.y + head.vOutBlock.dy * (headIdx - block.heads.length + 1);
            head.vOutBlock.y = lerp(head.vOutBlock.y, targetY, t_stack.t);
            head.vOutBlock.z = lerp(head.vOutBlock.z, targetZ, t_stack.t);
        }
    }

    let processInfo = startProcessBefore(state, block.attnOut);

    if (t_process.active) {
        processUpTo(state, t_process, block.attnOut, processInfo);
    }

    moveCameraTo(state, t_zoomOut, new Vec3(-8.304, 0.000, -175.482), new Vec3(293.606, 2.623, 2.618));

    if (t_processResid.active) {
        processUpTo(state, t_processResid, block.attnResidual, processInfo);
    }
}

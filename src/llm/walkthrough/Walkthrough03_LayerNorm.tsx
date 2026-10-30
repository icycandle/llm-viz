import React from "react";
import { findSubBlocks, splitGrid } from "../Annotations";
import { drawDataFlow } from "../components/DataFlow";
import { drawDependences } from "../Interaction";
import { clamp } from "@/src/utils/data";
import { lerp } from "@/src/utils/math";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";

export function walkthrough03_LayerNorm(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_str, c_blockRef, c_dimRef, breakAfter, cleanup } } = args;
    let { C } = layout.shape;

    if (wt.phase !== Phase.Input_Detail_LayerNorm) {
        return;
    }

    let ln = layout.blocks[0].ln1;

    setInitialCamera(state, new Vec3(-6.680, 0.000, -65.256), new Vec3(281.000, 9.000, 2.576));
    wt.dimHighlightBlocks = [layout.residual0, ...ln.cubes];

    commentary(wt, null, 0)`

The  ${c_blockRef('_input embedding_', state.layout.residual0)} matrix from the previous section is the input to our first Transformer block.

The first step in the Transformer block is to apply _layer normalization_ to this matrix. This is an
operation that normalizes the values in each column of the matrix separately.`.zh`

上一節的 ${c_blockRef('_輸入 embedding_', state.layout.residual0)} 矩陣，就是第一個 Transformer 區塊的輸入。

Transformer 區塊的第一步，是對這個矩陣套用 _layer normalization_（層正規化）。這個運算會把矩陣中每一欄的數值各自分開正規化。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t_hideExtra = afterTime(null, 1.0, 1.0);
    let t_moveInputEmbed = afterTime(null, 1.0);
    let t_moveCameraClose = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`
Normalization is an important step in the training of deep neural networks, and it helps improve the
stability of the model during training.

We can regard each column separately, so let's focus on the 4th column (${c_dimRef('t = 3', DimStyle.T)}) for now.`.zh`

正規化是訓練深度神經網路的重要步驟，有助於提升模型訓練時的穩定性。

每一欄可以分開處理，所以我們先專注在第 4 欄（${c_dimRef('t = 3', DimStyle.T)}）。`;

    breakAfter();
    let t_focusColumn = afterTime(null, 0.5);

    // mu ascii: \u03bc
    // sigma ascii: \u03c3
    breakAfter();
    commentary(wt)`
The goal is to make the average value in the column equal to 0 and the standard deviation equal to 1. To do this,
we find both of these quantities (${c_blockRef('mean (\u03bc)', ln.lnAgg1)} & ${c_blockRef('std dev (\u03c3)', ln.lnAgg2)}) for the column and then subtract the average and divide by the standard deviation.`.zh`

目標是讓這一欄的平均值等於 0、標準差等於 1。為此，我們先算出這一欄的這兩個量（${c_blockRef('平均值 (μ)', ln.lnAgg1)} 與 ${c_blockRef('標準差 (σ)', ln.lnAgg2)}），再減去平均值、除以標準差。`;

    breakAfter();

    let t_calcMuAgg = afterTime(null, 0.5);
    let t_calcVarAgg = afterTime(null, 0.5);

    // 1e-5 as 1x10^-5, but with superscript: 1x10<sup>-5</sup>

    breakAfter();
    commentary(wt)`
The notation we use here is E[x] for the average and Var[x] for the variance (of the column of length ${c_dimRef('C', DimStyle.C)}). The
variance is simply the standard deviation squared. The epsilon term (ε = ${<>1&times;10<sup>-5</sup></>}) is there to prevent division by zero.

We compute and store these values in our aggregation layer since we're applying them to all values in the column.

Finally, once we have the normalized values, we multiply each element in the column by a learned
${c_blockRef('weight (\u03b3)', ln.lnSigma)} and then add a ${c_blockRef('bias (β)', ln.lnMu)} value, resulting in our ${c_blockRef('normalized values', ln.lnResid)}.`.zh`

這裡的符號中，E[x] 表示平均值，Var[x] 表示變異數（針對長度為 ${c_dimRef('C', DimStyle.C)} 的這一欄）。變異數就是標準差的平方。epsilon 項（ε = ${<>1&times;10<sup>-5</sup></>}）是為了避免除以零。

因為這些值會套用到這一欄的所有數值，所以我們把它們算好後存放在彙總層（aggregation layer）中。

最後，得到正規化後的數值後，我們把這一欄的每個元素乘上一個學習得來的 ${c_blockRef('權重 (γ)', ln.lnSigma)}，再加上一個 ${c_blockRef('偏差 (β)', ln.lnMu)}，就得到 ${c_blockRef('正規化後的數值', ln.lnResid)}。`;

    breakAfter();

    let t_clean_aggs = afterTime(null, 0.2);
    cleanup(t_clean_aggs, [t_calcMuAgg, t_calcVarAgg]);
    let t_colSequence = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
We run this normalization operation on each column of the ${c_blockRef('input embedding matrix', layout.residual0)}, and the result is
the ${c_blockRef('normalized input embedding', ln.lnResid)}, which is ready to be passed into the Self-Attention layer.
`.zh`

我們對 ${c_blockRef('輸入 embedding 矩陣', layout.residual0)} 的每一欄都執行這個正規化運算，結果就是 ${c_blockRef('正規化後的輸入 embedding', ln.lnResid)}，準備好送進 self-attention 層。

`;

    breakAfter();
    let t_cleanupSplits = afterTime(null, 0.5);
    cleanup(t_cleanupSplits, [t_focusColumn]);
    if (t_cleanupSplits.t > 0) {
        t_colSequence.t = 0;
    }
    let t_runAggFull = afterTime(null, 2.0);
    let t_runNormFull = afterTime(null, 6.0);

    moveCameraTo(state, t_moveCamera, new Vec3(21.2, 0, -102.9), new Vec3(281.5, 11, 1.7));

    let exampleIdx = 3;
    let ln1 = layout.blocks[0].ln1;
    let inputBlock = layout.residual0;

    inputBlock.highlight = lerp(0, 0.3, t_hideExtra.t);

    let relevantBlocks = new Set([
        layout.residual0,
        ...ln1.cubes,
    ]);

    for (let blk of layout.cubes) {
            if (!relevantBlocks.has(blk)) {
            blk.opacity = lerp(1.0, 0.0, t_hideExtra.t);
        }
    }

    for (let blk of relevantBlocks) {
        if (blk != layout.residual0 && blk.t !== 'w') {
            blk.access!.disable = true;
        }
    }

    let startResidualY = layout.residual0.y;
    let endResidulY = ln1.lnResid.y;
    layout.residual0.y = lerp(startResidualY, endResidulY, t_moveInputEmbed.t);

    if (t_moveInputEmbed.t >= 0.0) {
        inputBlock.highlight = lerp(0.3, 0.0, t_moveInputEmbed.t);
    }

    moveCameraTo(state, t_moveCameraClose, new Vec3(-14.1, 0, -187.1), new Vec3(270, 4, 0.7));

    let splitAmt = lerp(0.0, 2.0, t_focusColumn.t);
    let splitPos = exampleIdx + 0.5;

    let otherColOpacity = lerp(1.0, 0.3, t_focusColumn.t);
    ln1.lnAgg1.opacity = otherColOpacity;
    ln1.lnAgg2.opacity = otherColOpacity;
    ln1.lnResid.opacity = otherColOpacity;
    inputBlock.opacity = otherColOpacity;

    if (t_focusColumn.t > 0) {
        let aggMuCol = splitGrid(layout, ln1.lnAgg1, Dim.X, splitPos, splitAmt)!;
        let aggVarCol = splitGrid(layout, ln1.lnAgg2, Dim.X, splitPos, splitAmt)!;
        let residCol = splitGrid(layout, ln1.lnResid, Dim.X, splitPos, splitAmt)!;
        let inputCol = splitGrid(layout, inputBlock, Dim.X, splitPos, splitAmt)!;
        aggMuCol.opacity = 1.0;
        aggVarCol.opacity = 1.0;
        residCol.opacity = 1.0;
        inputCol.opacity = 1.0;

        let aggDestIdx = new Vec3(exampleIdx, 0, 0);
        if (t_calcMuAgg.t > 0.0) {
            let pinIdx = new Vec3(0, 10, 0);
            drawDependences(state, ln1.lnAgg1, aggDestIdx);
            drawDataFlow(state, ln1.lnAgg1, aggDestIdx, pinIdx);
            aggMuCol.access!.disable = false;
            inputCol.highlight = 0.3;
        }

        if (t_calcVarAgg.t > 0.0) {
            let pinIdx = new Vec3(9, 9, 0);
            drawDependences(state, ln1.lnAgg2, aggDestIdx);
            drawDataFlow(state, ln1.lnAgg2, aggDestIdx, pinIdx);
            aggVarCol.access!.disable = false;
        }

        if (t_colSequence.t > 0.0) {
            aggMuCol.access!.disable = false;
            aggVarCol.access!.disable = false;

            let pinIdx = new Vec3(-10, 0, 0);

            let cPos = t_colSequence.t * C;
            let cIdx = clamp(Math.floor(cPos), 0, C - 1);
            let destIdx = new Vec3(exampleIdx, cIdx, 0);
            drawDependences(state, ln1.lnResid, destIdx);
            drawDataFlow(state, ln1.lnResid, destIdx, pinIdx);

            let targetCell = splitGrid(layout, residCol, Dim.Y, cIdx + 0.5, 0.0)!;
            targetCell.highlight = 0.3;

            findSubBlocks(residCol, Dim.Y, 0, cIdx).forEach((blk) => {
                blk.access!.disable = false;
            });
        }
    }

    if (t_runAggFull.t > 0.0) {
        try {
            let processInfo = startProcessBefore(state, ln1.lnAgg1);
            processUpTo(state, t_runAggFull, ln1.lnAgg2, processInfo);
            processUpTo(state, t_runNormFull, ln1.lnResid, processInfo);
        } catch (e) {
            console.log(e);
        }
    }
}

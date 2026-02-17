"""
Expert Blending Model: DNNBackbone + DNNAdapter + NNUEExperts

局面に応じたNNUE評価関数を動的に合成する。
dlshogiのResNet backboneで局面特徴を抽出し、adapterでN_EXPERTS個のNNUE重みの
混合比を決定、加重平均した重みでNNUE評価を行う。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.network.policy_value_network_resnet10_swish import (
    PolicyValueNetwork as DlshogiPolicyValueNetwork,
)
from dlshogi import serializers


class DNNBackbone(nn.Module):
    """dlshogiのPolicyValueNetworkからbackbone部分(10 residual blocks)の出力を取り出すラッパー。

    入力: x1 (batch, 62, 9, 9), x2 (batch, FEATURES2_NUM, 9, 9)
    出力: feat (batch, 192, 9, 9) — u21に相当（policy/valueヘッド前の最終residual block出力）

    重みはrequires_grad=Falseで固定し、BatchNormは常にevalモードで動作する。
    """

    def __init__(self, model: DlshogiPolicyValueNetwork):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, x1, x2):
        """backbone forwardのみ実行し、u21 (batch, 192, 9, 9) を返す。"""
        m = self.model
        # Input projection
        u1_1_1 = m.l1_1_1(x1)
        u1_1_2 = m.l1_1_2(x1)
        u1_2 = m.l1_2(x2)
        u1 = m.swish(m.norm1(u1_1_1 + u1_1_2 + u1_2))
        # Residual block 1
        h2 = m.swish(m.norm2(m.l2(u1)))
        u3 = m.swish(m.norm3(m.l3(h2)) + u1)
        # Residual block 2
        h4 = m.swish(m.norm4(m.l4(u3)))
        u5 = m.swish(m.norm5(m.l5(h4)) + u3)
        # Residual block 3
        h6 = m.swish(m.norm6(m.l6(u5)))
        u7 = m.swish(m.norm7(m.l7(h6)) + u5)
        # Residual block 4
        h8 = m.swish(m.norm8(m.l8(u7)))
        u9 = m.swish(m.norm9(m.l9(h8)) + u7)
        # Residual block 5
        h10 = m.swish(m.norm10(m.l10(u9)))
        u11 = m.swish(m.norm11(m.l11(h10)) + u9)
        # Residual block 6
        h12 = m.swish(m.norm12(m.l12(u11)))
        u13 = m.swish(m.norm13(m.l13(h12)) + u11)
        # Residual block 7
        h14 = m.swish(m.norm14(m.l14(u13)))
        u15 = m.swish(m.norm15(m.l15(h14)) + u13)
        # Residual block 8
        h16 = m.swish(m.norm16(m.l16(u15)))
        u17 = m.swish(m.norm17(m.l17(h16)) + u15)
        # Residual block 9
        h18 = m.swish(m.norm18(m.l18(u17)))
        u19 = m.swish(m.norm19(m.l19(h18)) + u17)
        # Residual block 10
        h20 = m.swish(m.norm20(m.l20(u19)))
        u21 = m.swish(m.norm21(m.l21(h20)) + u19)
        return u21

    def train(self, mode=True):
        """BatchNormを常にevalモードに保つ。"""
        super().train(mode)
        self.model.eval()
        return self


class DNNAdapter(nn.Module):
    """backboneの特徴マップからexpert混合重みを計算するゲーティングネットワーク。

    入力: feat (batch, 192, 9, 9)
    処理: Global Average Pooling → FC → ReLU → FC → [noise +] softmax
    出力: weights (batch, N_EXPERTS), 総和=1
    """

    def __init__(self, in_channels=192, hidden_dim=128, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)

    def forward(self, feat, training=True):
        """
        Args:
            feat: (batch, C, H, W) backbone特徴マップ
            training: Trueの場合、logitsにGaussian noiseを加える
        Returns:
            weights: (batch, N_EXPERTS) softmax正規化された混合重み
        """
        # Global Average Pooling: (batch, C, H, W) -> (batch, C)
        x = feat.mean(dim=[2, 3])
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        if training:
            noise = torch.randn_like(logits)
            logits = logits + noise
        weights = F.softmax(logits, dim=-1)
        return weights


class NNUEExperts(nn.Module):
    """N_EXPERTS個のNNUE重みセットを保持し、加重平均して1つのNNUEとしてforward計算する。

    NNUE構造:
        input: Linear(num_features, 256) — bias有り
        l1: Linear(512, 32)  — 両視点を結合(2*256=512)
        l2: Linear(32, 32)
        output: Linear(32, 1)
        活性化: clipped ReLU (clamp 0.0〜1.0)

    各レイヤーの重み・バイアスを (N_EXPERTS, *param_shape) のParameterとして保持し、
    gate_weights で「重みを先に合成してから推論」する。
    """

    L1 = 256
    L2 = 32
    L3 = 32

    def __init__(self, n_experts, num_features):
        super().__init__()
        self.n_experts = n_experts
        self.num_features = num_features

        # Expert weight parameters: (N_EXPERTS, out_features, in_features)
        self.input_weight = nn.Parameter(torch.zeros(n_experts, self.L1, num_features))
        self.input_bias = nn.Parameter(torch.zeros(n_experts, self.L1))

        self.l1_weight = nn.Parameter(torch.zeros(n_experts, self.L2, 2 * self.L1))
        self.l1_bias = nn.Parameter(torch.zeros(n_experts, self.L2))

        self.l2_weight = nn.Parameter(torch.zeros(n_experts, self.L3, self.L2))
        self.l2_bias = nn.Parameter(torch.zeros(n_experts, self.L3))

        self.output_weight = nn.Parameter(torch.zeros(n_experts, 1, self.L3))
        self.output_bias = nn.Parameter(torch.zeros(n_experts, 1))

    def _blended_linear(self, x, weight, bias, gate_weights):
        """線形層を expert 重みでブレンドして計算する。

        Args:
            x: (batch, in_features)
            weight: (n_experts, out_features, in_features)
            bias: (n_experts, out_features)
            gate_weights: (batch, n_experts)
        Returns:
            out: (batch, out_features)
        """
        out = None
        for k in range(self.n_experts):
            out_k = F.linear(x, weight[k], bias[k])  # (batch, out_features)
            g = gate_weights[:, k].unsqueeze(1)      # (batch, 1)
            out = out_k * g if out is None else out + (out_k * g)
        return out

    def forward(self, gate_weights, us, them, w_in, b_in):
        """
        gate_weights で各層の重みを合成した 1 つの NNUE として forward 計算する。
        本番対局時 (`blend_expert_weights`) と同じ意味論に合わせる。

        Args:
            gate_weights: (batch, N_EXPERTS) expert混合重み
            us: (batch, 2*L1) 手番側の視点マスク
            them: (batch, 2*L1) 相手側の視点マスク
            w_in: (batch, num_features) 白視点のスパース入力特徴
            b_in: (batch, num_features) 黒視点のスパース入力特徴
        Returns:
            output: (batch, 1) NNUE評価値
        """
        # sparse tensor → dense 変換
        if w_in.is_sparse:
            w_in = w_in.to_dense()
        if b_in.is_sparse:
            b_in = b_in.to_dense()

        # Input layer:
        # linear は重みに対して線形なので、重みブレンド後の F.linear と等価に
        # Σ_k gate_k * F.linear(x, W_k, b_k) で計算できる。
        # これにより (batch, 256, num_features) の巨大中間テンソル生成を避ける。
        w = self._blended_linear(w_in, self.input_weight, self.input_bias, gate_weights)
        b = self._blended_linear(b_in, self.input_weight, self.input_bias, gate_weights)

        # 視点の結合
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)

        # Hidden layers and output: blended-weight semantics
        l1_ = self._blended_linear(l0_, self.l1_weight, self.l1_bias, gate_weights)
        l1_ = torch.clamp(l1_, 0.0, 1.0)

        l2_ = self._blended_linear(l1_, self.l2_weight, self.l2_bias, gate_weights)
        l2_ = torch.clamp(l2_, 0.0, 1.0)

        output = self._blended_linear(
            l2_, self.output_weight, self.output_bias, gate_weights
        )

        return output


class ExpertBlendingModel(nn.Module):
    """DNNBackbone + DNNAdapter + NNUEExpertsを統合したモデル。

    forward計算:
        1. backbone (frozen) で局面特徴を抽出
        2. adapter で expert混合重みを計算
        3. nnue_experts で加重平均したNNUE重みによる評価値を計算
    """

    def __init__(self, backbone, adapter, nnue_experts):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.nnue_experts = nnue_experts

    def forward(self, x1, x2, us, them, w_in, b_in, training=True):
        """
        Args:
            x1: (batch, 62, 9, 9) dlshogi features1
            x2: (batch, FEATURES2_NUM, 9, 9) dlshogi features2
            us: (batch, 512) 手番側の視点マスク
            them: (batch, 512) 相手側の視点マスク
            w_in: (batch, num_features) 白視点のNNUEスパース入力
            b_in: (batch, num_features) 黒視点のNNUEスパース入力
            training: adapter noiseの有無
        Returns:
            value: (batch, 1) NNUE評価値
        """
        with torch.no_grad():
            feat = self.backbone(x1, x2)
        gate_weights = self.adapter(feat, training=training)
        value = self.nnue_experts(gate_weights, us, them, w_in, b_in)
        return value


def load_backbone(weights_path, device='cpu'):
    """dlshogiの学習済み重みを読み込んでDNNBackboneを返す。

    Args:
        weights_path: dlshogi .npz 重みファイルパス
        device: デバイス
    Returns:
        DNNBackbone (frozen, eval mode)
    """
    model = DlshogiPolicyValueNetwork()
    serializers.load_npz(weights_path, model)
    backbone = DNNBackbone(model)
    backbone.to(device)
    return backbone


def load_nnue_experts(ckpt_path, n_experts, feature_set):
    """NNUEチェックポイントからN_EXPERTS個のexpert重みを初期化する。

    学習済みの1つのNNUE重みをN_EXPERTS個に複製して初期化する。

    Args:
        ckpt_path: PyTorch Lightning .ckpt ファイルパス
        n_experts: expert数
        feature_set: HalfKP等のFeatureBlockインスタンス
    Returns:
        NNUEExperts
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    num_features = feature_set.num_features
    experts = NNUEExperts(n_experts, num_features)

    # NNUE state_dict key → NNUEExperts attribute name
    param_map = {
        'input.weight': 'input_weight',
        'input.bias': 'input_bias',
        'l1.weight': 'l1_weight',
        'l1.bias': 'l1_bias',
        'l2.weight': 'l2_weight',
        'l2.bias': 'l2_bias',
        'output.weight': 'output_weight',
        'output.bias': 'output_bias',
    }

    with torch.no_grad():
        for nnue_key, expert_key in param_map.items():
            src = state_dict[nnue_key]  # (out_features, in_features) or (out_features,)
            dst = getattr(experts, expert_key)  # (n_experts, ...)
            # input層の重みサイズが異なる場合 (HalfKP → HalfKP^ のパディング)
            if src.shape != dst.shape[1:]:
                padded = torch.zeros(dst.shape[1:])
                # src の範囲だけコピーし、残り (virtual features) はゼロ
                slices = tuple(slice(0, s) for s in src.shape)
                padded[slices] = src
                src = padded
            # 1つのNNUE重みをN_EXPERTS個に複製
            dst.copy_(src.unsqueeze(0).expand_as(dst))

    return experts


def create_expert_blending_model(backbone_weights_path, nnue_ckpt_path, feature_set,
                                 n_experts=4, adapter_hidden=128, device='cpu'):
    """全コンポーネントを組み立ててExpertBlendingModelを返すファクトリ関数。

    Args:
        backbone_weights_path: dlshogi .npz 重みファイルパス
        nnue_ckpt_path: NNUE PyTorch Lightning .ckpt ファイルパス
        feature_set: HalfKP等のFeatureBlockインスタンス
        n_experts: expert数
        adapter_hidden: adapter中間層の次元数
        device: デバイス
    Returns:
        ExpertBlendingModel
    """
    backbone = load_backbone(backbone_weights_path, device)
    adapter = DNNAdapter(in_channels=192, hidden_dim=adapter_hidden, n_experts=n_experts)
    nnue_experts = load_nnue_experts(nnue_ckpt_path, n_experts, feature_set)

    model = ExpertBlendingModel(backbone, adapter, nnue_experts)
    model.to(device)
    return model

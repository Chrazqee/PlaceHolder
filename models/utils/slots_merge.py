import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


# [ ]: 聚类不可微分的问题!
class SlotClustering:
    def __init__(self, num_clusters=5, normalize=True):
        self.num_clusters = num_clusters
        self.normalize = normalize

    def __call__(self, slots):
        # slots: [B, S, D]
        B, S, D = slots.shape
        clustered_slots = []

        for b in range(B):
            slot_b = slots[b]  # [S, D]
            if self.normalize:
                slot_b = F.normalize(slot_b, dim=-1)

            # 转为 numpy 做 KMeans
            slot_np = slot_b.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
            labels = kmeans.fit_predict(slot_np)

            # 聚合每个 cluster 的 slot 表征
            slot_tensor = torch.tensor(slot_np, device=slots.device)
            cluster_means = []
            for k in range(self.num_clusters):
                mask = torch.tensor(labels == k, device=slots.device)
                if mask.sum() == 0:
                    cluster_means.append(torch.zeros(D, device=slots.device))
                else:
                    cluster_means.append(slot_tensor[mask].mean(dim=0))
            clustered_slots.append(torch.stack(cluster_means))  # [K, D]

        return torch.stack(clustered_slots)  # [B, K, D]

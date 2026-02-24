# Title: Semi-Supervised Adaptive Learning for Streaming Data in Internet of Things Applications
-------------------------------------------------------------------------------------------------------------------------------
Abstract
-------------------------------------------------------------------------------------------------------------------------------
Classifying data streams in real-world settings is challenging due to three interconnected problems: concept drift, dynamic class imbalance, and label scarcity. While many methods handle these problems individually, their co-occurrence in domains such as Internet of Things (IoT) monitoring or network security highlights a major limitation in stream learning. The majority of existing approaches rely on the assumption that fully labeled data is available, which is rarely practical in real-world deployments. To overcome these challenges, we propose a new semi-supervised learning model based on an adaptive nearest neighbor approach for evolving data streams. We employ adaptive knowledge windows that use temporal fading and performance-based weighting. For every class, a separate knowledge window is kept which ensures a balanced and memory efficient summary of recent data points. Additionally, a three-phase active learning approach is used to find the most informative data points for labeling by utilizing class-balance awareness, random exploration, and uncertainty sampling. Lastly, based on ongoing predictive performance, a validation-based nearest neighbor classifier dynamically adjusts instance relevance and neighborhood size. Comprehensive experiments on 32 diverse datasets, including both real-world and synthetic datasets with varying drifts, demonstrate that our method consistently outperforms state-of-the-art supervised and semi-supervised techniques. Especially, proposed method uses only 
 of labeled data and achieves competitive performance with fully supervised methods. In summary, the proposed method advances the field of artificial intelligence (AIoT) by providing a resource-efficient and lightweight solution suitable for real-time deployment in sensor networks, edge computing platforms, and IoT systems.
 
-------------------------------------------------------------------------------------------------------------------------------

This is the version 1, and it will be constantly improved. We will update the progress.

-------------------------------------------------------------------------------------------------------------------------------

Reference: Salah Uddin, Muhammad Abbas, Kun Huang, Ebenezer Nanor, "Semi-Supervised Adaptive Learning for Streaming Data in Internet of Things Applications," in Internet of Things Feb. 2026, doi: [10.1109/TCYB.2021.3070420](https://doi.org/10.1016/j.iot.2026.101898).

-------------------------------------------------------------------------------------------------------------------------------

ATTN: This code were developed by Salah UdUin (salahuddin@zyufl.edu.cn). For any problem and suggestment, please feel free to contact Dr. Salah Ud Din.

import torch
anchor_sizes1 = ((20,600), (50,500), (60,100), (20,120), (25,500),(30,100),
                            (10,300), (25,250), (30,50), (10,60), (13,250),(15,50))
anchor_sizes2 = ((32,), (64,), (128,), (256,), (512,))

aspect_ratios1 = ((1.0),) * len(anchor_sizes1)
aspect_ratios2 = ((0.5, 1.0, 2.0),) * len(anchor_sizes2)

# for sizes, aspect_ratios in zip(anchor_sizes1, aspect_ratios1):
#     print(sizes, aspect_ratios )

def generate_anchors(scales,
                     aspect_ratios,
                     ):
    # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
    """
    compute anchor sizes
    Arguments:
        scales: sqrt(anchor_area)
        aspect_ratios: h/w ratios
        dtype: float32
        device: cpu/gpu
    """
    scales = torch.as_tensor(scales )
    aspect_ratios = torch.as_tensor(aspect_ratios)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    # [r1, r2, r3]' * [s1, s2, s3]
    # number of elements is len(ratios)*len(scales)
    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)

    # print(ws,ws.shape)  # shape : 75
    # print(hs,hs.shape)  # shape : 75
    print([-ws, -hs, ws, hs])
    print(torch.stack([-ws, -hs, ws, hs], dim=1))
    # left-top, right-bottom coordinate relative to anchor center(0, 0)
    # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
    # base_anchors = torch.stack([-ws[0], -hs[0], ws[1], hs[1]], dim=1) / 2
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    return base_anchors.round()  # round 四舍五入

# print(generate_anchors(anchor_sizes2,aspect_ratios2),generate_anchors(anchor_sizes2,aspect_ratios2).shape)
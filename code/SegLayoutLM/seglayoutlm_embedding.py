from torch import nn as nn
import torch
from .seglayoutlm_configuration import SegLayoutLMConfig

class SegLayoutLMEmbedding(nn.Module):
    """
    """
    def __init__(self, config:SegLayoutLMConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_position_embedding = nn.Embedding(config.max_token_position_embeddings, config.hidden_size)
        self.seg_position_embedding = nn.Embedding(config.max_seg_position_embeddings, config.hidden_size)
        self.token_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.x_position_embedding = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embedding = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _calc_seg_text_embeddings(self, input_ids, segment_ids):
        """

        Args:
            input_ids: token ids
            segment_ids: segment id corresponding to each token id
        Returns:
            Average the token embeddings inside segments to get the segment embeddings.

        """
        max_seg_size, hidden_size = self.seg_position_embedding.shape[0], self.seg_position_embedding.shape[1]
        masks = []
        for i in range(segment_ids.shape[0]):
            mul_matrix = torch.zeros((max_seg_size, hidden_size))
            mul_matrix[segment_ids[i], torch.arange(hidden_size)] = 1
            mul_matrix = nn.functional.normalize(mul_matrix, p=1, dim=1)
            masks.append(mul_matrix)
        masks = torch.stack(masks)
        word_embeds = self.word_embeddings(input_ids)
        return torch.bmm(masks, word_embeds)


    def _calc_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings


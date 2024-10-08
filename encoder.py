import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size : int, heads : int):
        '''
            embed_size : maintained both in encoder and decoder
        '''
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.vlinear = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias = False) for i in range(heads)])
        self.klinear = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias = False) for i in range(heads)])
        self.qlinear = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias = False) for i in range(heads)])

        self.feedforward = nn.Linear(embed_size, embed_size, bias = False)

    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, mask : torch.Tensor):
        '''
            current shapes:
            query, key, value : (batch_dates, num_companies, embed_size ) || (batch_dates, num_companies, heads, head_dim)
            mask : (batch_dates, num_companies)
        '''
        ## query size = (N, length, embed_size)
        ## paralle size = N, len, heads, head_dim
        queries_parallel = query.reshape(query.shape[0], query.shape[1], self.heads, self.head_dim)
        keys_parallel = key.reshape(key.shape[0], key.shape[1], self.heads, self.head_dim)
        values_parallel = value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim )
        
        Scaled_Attention = []

        for i in range(self.heads):
            query_slice = self.qlinear[i](queries_parallel[:,:,i]) ## shape = N, qlen, head_dim
            key_slice = self.klinear[i](keys_parallel[:,:,i]) ## shape = N, klen, head_dim
            value_slice = self.vlinear[i](values_parallel[:,:,i]) ## shape = N, vlen, head_dim
            qk_product = torch.einsum("nqd,nkd->nqk",[query_slice, key_slice]) / (self.embed_size**(1/2))
            # qk_product = torch.einsum("nqd,nqk->ndk",[query_slice, key_slice]) / (self.embed_size**(1/2))
            if mask is not None:
                # exp_mask = torch.unsqueeze(mask, dim = 2) 
                qk_product =  qk_product.masked_fill(mask == torch.tensor(0), float("-1e28"))
            qk_probs = torch.softmax(qk_product, dim = 2)
            attention = torch.einsum("nqk,nkh->nqh",[qk_probs, value_slice])
            # attention = torch.einsum("ndk,nqd->nqk",[qk_probs, value_slice])
            # print(attention.shape)
            Scaled_Attention.append(attention)
        # stack along the heads
        Scaled_Attention = torch.stack(Scaled_Attention,dim = 2).reshape(query.shape[0], query.shape[1],-1)
        out = self.feedforward(Scaled_Attention)
        return out 
    

class BlockLayer(nn.Module):
    def __init__(self, embed_size : int, inner_dim : int, heads : int, dropout : int):
        super(BlockLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_size, heads)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

        self.feedforward = nn.Sequential(
                nn.Linear(embed_size, inner_dim, bias = True),
                nn.ReLU(),
                nn.Linear(inner_dim, embed_size, bias = True)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        out1 = self.dropout(self.attention(query, key, value, mask))
        out1 = self.layernorm1(out1 + query)

        output = self.dropout(self.feedforward(out1))
        output = self.layernorm2(output + out1)
        return output
    

class Encoder(nn.Module):
    def __init__(self, embed_size : int, num_companies : int, num_characteristics : int, inner_dim : int,  heads : int, repeats : int, dropout : float):
        super(Encoder, self).__init__()
        # self.company_embeds = nn.Embedding(num_companies+1, embed_size)
        self.characteristics_embeds = nn.Linear(num_characteristics, embed_size)
        self.num_characteristics = num_characteristics

        self.dropout = nn.Dropout(dropout)
        self.Transformer_Blocks = nn.ModuleList([BlockLayer(embed_size, inner_dim, heads, dropout) for i in range(repeats)])

    def forward(self, company_characteristics : torch.Tensor, company_mask : torch.BoolTensor = None):
        '''
            Company_ids : Will contain the ids starting from 1 to number_of_companies.
            Company_characteristics : Will contain the numerical characteristics of the companies returns.
            Company_mask : Only use it during company id prediction training paradigm 1.
            Training : {
                1. If the company_mask is not None =>  Predict the company_id. Here Softmax Loss is applied.
                2. If the return_mask is not None => Predict the return. Here MLE loss is applied.
            }  

            Company_type_tensor : (Future addition) Company type embedders 

            Expected Shapes: 
            Company_ids : (batch_of_dates, num_companies)
            Company_Characteristics : (batch _of_dates, num_companies, num_company_characteristics)
            company_mask : (batch_dates, 1)

            Propositions:
            company_mask : (batch_dates, num_companies)
            charac_embeds : (batch_dates, num_companies, embed_size)
            company_embeds : (batch_dates, num_companies, embed_size)
        '''
        # batch_dates = company_ids.shape[0]
        charac_embeds = self.characteristics_embeds(company_characteristics) ### (batch_dates, num_companies, num_company_characteristics, embed_size)
        # company_embeds = self.company_embeds(company_ids) ### (batch_dates, num_companies, embed_size)
        # company_embeds = torch.unsqueeze(company_embeds, dim = 2) ### (batch_dates, num_companies, 1, embed_size)
        # print(f"Characteristics Embedding shape = {charac_embeds.shape}")
        # print(f"Company Embedding shape = {company_embeds.shape}")
        encoded = self.dropout(charac_embeds)

        for transformer_block in self.Transformer_Blocks:
            encoded = transformer_block(encoded, encoded, encoded, company_mask)
        return encoded
    
class EncFinTransformer(nn.Module):
    def __init__(self, embed_size : int, inner_dim : int, num_companies : int, num_characteristics : int, heads : int, repeats : int, dropout : float, mode : str = "return") -> None:
        super().__init__()

        self.encoder = Encoder(embed_size, num_companies, num_characteristics, inner_dim, heads, repeats, dropout)
        self.final_layer = nn.Linear(embed_size, 1, bias = True)

    
    def forward(self, company_characteristics, return_inputs, company_mask, return_mask):

        encoder_output = self.encoder(company_characteristics, company_mask)
        # print(f"Encoder output shape = {encoder_output.shape}")
        output = self.final_layer(encoder_output)
        # print(output.shape)
        # print(f"Output shape = {output.shape}")
        return output 


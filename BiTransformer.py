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
            
            mask : (batch, num_characteristics)
            qk_product : (batch, qlen, klen)
        '''
        ## query size = (N, length, embed_size)
        ## parallel size = N, len, heads, head_dim

        batch_size = query.shape[0]

        queries_parallel = query.reshape(query.shape[0], query.shape[1], self.heads, self.head_dim)
        keys_parallel = key.reshape(key.shape[0], key.shape[1], self.heads, self.head_dim)
        values_parallel = value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim )
        
        Scaled_Attention = []

        for i in range(self.heads):
            query_slice = self.qlinear[i](queries_parallel[:,:,i]) ## shape = N, qlen, head_dim
            key_slice = self.klinear[i](keys_parallel[:,:,i]) ## shape = N, klen, head_dim
            value_slice = self.vlinear[i](values_parallel[:,:,i]) ## shape = N, vlen, head_dim
            qk_product = torch.einsum("nqd,nkd->nqk",[query_slice, key_slice]) / (self.embed_size**(1/2))

            if mask is not None:
                attention_mask = torch.zeros_like(qk_product)
                q_unattention = torch.FloatTensor([float('-1e16') for _ in range(query.shape[1])])
                k_unattention = torch.FloatTensor([float('-1e16') for _ in range(key.shape[1])])

                for batch_num in range(batch_size):
                    un_attention_indices = torch.where(mask[i] == -torch.inf, True, False) ### un_attention_indices = (indices)
                    # print(f"Un attention Indices shape = {un_attention_indices}")
                    for index in range(un_attention_indices.shape[0]):
                        if un_attention_indices[index] == True:
                            attention_mask[batch_num][:,index] = k_unattention 
                            attention_mask[batch_num][index,:] = q_unattention
                # print(f"Attention Mask = {attention_mask}")
                qk_product = qk_product + attention_mask
            
            qk_probs = torch.softmax(qk_product, dim = 2)
            # print(qk_probs)
            attention = torch.einsum("nqk,nkh->nqh",[qk_probs, value_slice])
            # print(attention)
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
    def __init__(self, embed_size : int, num_characteristics : int, inner_dim : int,  heads : int, repeats : int, dropout : float):
        super().__init__()

        ### Characteristics Embedding layer
        # self.columns_embedder = nn.Embedding(num_embeddings=94, embedding_dim=embed_size)

        ### We produce characteristic embeds for each characteristics
        self.characteristics_embeds = nn.ModuleList([nn.Linear(1, embed_size) for _ in range(num_characteristics)])
        self.num_characteristics = num_characteristics

        self.dropout = nn.Dropout(dropout)
        self.Transformer_Blocks = nn.ModuleList([BlockLayer(embed_size, inner_dim, heads, dropout) for i in range(repeats)])


    def forward(self, company_characteristics : torch.Tensor, attention_mask : torch.FloatTensor = None):
        """
            company_characteristics : (BatchSize, 94)
            attention_mask : (BatchSize, 94)
            charac_embeddings : (BatchSize, )
        """
        company_characteristics = company_characteristics + attention_mask
        charac_emebeddings = []

        for i in range(self.num_characteristics):
            # pos_embed = self.columns_embedder(i)
            single_embed = self.characteristics_embeds[i](company_characteristics[:,i].unsqueeze(dim = 1)) ### company_characteristics[:,i] : (B)

            ### single_embed : (B, embed_size)
            charac_emebeddings.append(single_embed) ### May be add position Embeddings

        charac_emebeddings = torch.stack(charac_emebeddings, dim = 1) ### charac_embeddings : (B, num_characs, embed_size)
        encoded = self.dropout(charac_emebeddings)

        for transformer_block in self.Transformer_Blocks:
            encoded = transformer_block(encoded, encoded, encoded, attention_mask)
        return encoded
    


class BetaTransformer(nn.Module):
    def __init__(self, embed_size : int, inner_dim : int, output_dim : int,
                 num_characteristics : int, heads : int, repeats : int, dropout : float):
        super().__init__()
        self.num_characteristics = num_characteristics

        # self.linear_out = nn.Linear(num_characteristics, output_dim)
        self.linear_out = nn.ModuleList([nn.Linear(embed_size,1) for _ in range(num_characteristics)])
        self.beta_linear = nn.Linear(num_characteristics, output_dim)
        self.encoder = Encoder(embed_size, num_characteristics, inner_dim, heads, repeats, dropout)


    def forward(self, company_characteristics : torch.FloatTensor, mask : torch.LongTensor):
        encoder_output = self.encoder(company_characteristics, mask) ### Encoder output
        outs = []

        for i in range(self.num_characteristics):
            outs.append(self.linear_out[i]( encoder_output[:,i]))

        outs = torch.cat(outs, dim = 1)
        outs = self.beta_linear(outs)

        return outs
    


class FactorTransformer(nn.Module):
    def __init__(self, embed_size : int, inner_dim : int, output_dim : int,
                 num_characteristics : int, heads : int, repeats : int, dropout : float):
        super().__init__()
        self.num_characteristics = num_characteristics

        self.linear_out = nn.ModuleList([nn.Linear(embed_size,1) for _ in range(num_characteristics)])
        self.encoder = Encoder(embed_size, num_characteristics, inner_dim, heads, repeats, dropout)

        self.lin1 = nn.Linear(num_characteristics,output_dim)


    def forward(self, company_characteristics : torch.FloatTensor, target_returns : torch.FloatTensor, mask : torch.LongTensor):

        #### First the encoder interaction ####
        encoder_output = self.encoder(company_characteristics, mask) #### Shape : [ N, num_characs, embed_size ]
        outs = []

        for i in range(self.num_characteristics):
            outs.append(self.linear_out[i](encoder_output[:,i]))

        outs = torch.cat(outs, dim = 1)


        #### Managed encoder outputs ####
        managed_encoder_outputs = torch.mm(outs.T, outs) # P X P
        managed_encoder_outputs = torch.inverse(managed_encoder_outputs) # P X P
        managed_encoder_outputs = torch.mm(managed_encoder_outputs, outs.T) # P X N 
        managed_encoder_outputs = torch.mm(managed_encoder_outputs, target_returns) # P X 1
        managed_encoder_outputs = managed_encoder_outputs.squeeze(dim = 1)

        #### Extract the factors from them
        factors = self.lin1(managed_encoder_outputs)

        return factors
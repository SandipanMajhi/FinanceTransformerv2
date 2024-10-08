'''
    Latest to do:
    1. Fix encoder embeddings (done)
    2. Fix Masked_Fill (done)
    3. Final Transformer implementation. (done)

    Future to do:
    1. Multiple Mask ids in company and returns (returns done)


'''

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
            if mask is not None:
                exp_mask = torch.unsqueeze(mask, dim = 2) 
                # print(f"Mask Shape = {mask.shape}")
                # print(f"qk product shape = {qk_product.shape}")
                qk_product =  qk_product.masked_fill(exp_mask == torch.tensor(0), float("-1e28"))
            qk_probs = torch.softmax(qk_product, dim = 2)
            attention = torch.einsum("nqk,nkh->nqh",[qk_probs, value_slice])
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
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, inner_dim, heads, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(embed_size, heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.encoder_decoder_attention = BlockLayer(embed_size, inner_dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_out, return_mask = None, company_mask = None):
        decoder_attention_out = self.dropout(self.masked_attention(trg, trg, trg, return_mask))
        decoder_attention_out = self.layer_norm1(decoder_attention_out + trg)
        enc_dec_attention = self.encoder_decoder_attention(decoder_attention_out, enc_out, enc_out, company_mask)
        return enc_dec_attention
    
class Decoder(nn.Module):
    def __init__(self, embed_size : int, num_companies : int, inner_dim : int,  heads : int, repeats : int, dropout : int):
        super(Decoder, self).__init__()
        self.return_encoding = nn.Linear(1, embed_size)
        self.num_returns = num_companies
        self.embed_size = embed_size

        self.Decoder_Blocks = nn.ModuleList([DecoderBlock(embed_size, inner_dim, heads, dropout) for i in range(repeats)])
        self.final_linear_layer = nn.Linear(embed_size, 1, bias = False)
        self.return_predictor = nn.Linear(num_companies, 1, bias = True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out : torch.Tensor, return_inputs : torch.Tensor, return_mask : torch.BoolTensor = None, company_mask = None):
        '''
            enc_out :  This is the attended output of the encoder
            return_inputs : The returns for the companies in the inputs
            Training : {
                1. If the company_mask is not None =>  Predict the company_id. Here Softmax Loss is applied. Here,
                return_mask is None.
                2. If the return_mask is not None => Predict the return. Here MLE loss is applied. Return_mask not None.
            }  

            return_mask : masks the ids in where the return is to be calculated. Transforms the encoding into a zero vector for
            that ID.

            Shapes:
            return_inputs : (batch_dates, num_companies, 1)
            return_mask : (batch_dates, 1)  
            return_encodings : (batch_dates, num_companies, embed_size)

            Current Propositions:
            Pros : Will help in implementing masked_fill:
            return_mask : (batch_dates, num_companies) ### a boolean tensor
            company_mask ; (batch_dates, num_companies) ### a boolean tensor
        '''
        num_batches = return_inputs.shape[0]
        return_encodings = self.return_encoding(return_inputs) #### (batch_dates, num_companies, embed_size)

        ##### Not Needed ######
        # if return_mask is not None:
        #     return_mask = torch.unsqueeze(return_mask, dim = 2) 
        #     return_encodings = return_encodings.masked_fill(return_mask == torch.tensor(0), "1e-28")
        ##### Not Needed ######

        decoder_outs = self.dropout(return_encodings)

        for decoder_block in self.Decoder_Blocks:
            decoder_outs = decoder_block(decoder_outs, enc_out, return_mask, company_mask)
        
        outputs = self.final_linear_layer(decoder_outs)
        # outputs = self.return_predictor(outputs)
        return outputs


class FinTransformer(nn.Module):
    def __init__(self, embed_size : int, inner_dim : int, num_companies : int, num_characteristics : int, heads : int, repeats : int, dropout : float, mode : str = "return"):
        super(FinTransformer, self).__init__()
        self.encoder = Encoder(embed_size, num_companies, num_characteristics, inner_dim, heads, repeats, dropout)
        self.decoder = Decoder(embed_size, num_companies, inner_dim, heads, repeats, dropout)

        self.mode = mode
        self.num_companies = num_companies


    def forward(self, company_characteristics : torch.FloatTensor,
                 return_inputs : torch.FloatTensor, company_mask : torch.LongTensor, 
                 return_mask : torch.LongTensor):
        '''
        Constraints:
            Mask_token_id = 0 
            company_mask_id = [1, num_companies]
            return_mask_id = [1, num_companies]

        Current Proposition inputs:
            (Single Mask ids)
            company_mask_ids : (batch_dates, 1)
            return_mask_ids : (batch_dates, 1)

        outputs : 
        company_mask = (batch_dates, num_companies)
        return_mask = (batch_dates, num_companies)
        return_inputs : (batch_dates, num_companies, 1)
        '''
        # self.num_batches = company_ids.shape[0]
        # company_mask = self.create_company_mask(company_mask_ids)
        # return_mask = self.create_return_mask(return_mask_ids)

        # print(f"Company mask shape = {company_mask.shape}")
        # print(f"Return Mask shape = {return_mask.shape}")

        encoder_output = self.encoder(company_characteristics, company_mask)
        decoder_output = self.decoder(encoder_output, return_inputs, return_mask, company_mask)

        return decoder_output


    def create_company_mask(self, company_mask_ids : torch.LongTensor):
        company_mask = torch.ones(self.num_batches, self.num_companies)
        if company_mask_ids is not None:
            for batch in range(self.num_batches):
                company_mask[batch, company_mask_ids[batch]] = 0
        return company_mask.to(device)


    def create_return_mask(self, return_mask_ids : torch.LongTensor):
        return_mask = torch.ones(self.num_batches, self.num_companies)
        if return_mask_ids is not None:
            for batch in range(self.num_batches):
                return_mask[batch, return_mask_ids[batch]] = 0
        return return_mask.to(device)


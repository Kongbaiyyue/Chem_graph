import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from functools import partial

from molbart.models.util import (
    PreNormEncoderLayer,
    PreNormDecoderLayer,
    FuncLR,
    PreNormCrossLayer,
    PreNormCross,
    PreNormDecoder
)
from molbart.models.graph_transformer_pytorch import (
    GraphTransformer,
    TextTransformer,
    Attention,
    GraphCrossformer
)


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Abstract Models ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class _AbsTransformerModel(pl.LightningModule):
    def __init__(
        self,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule,
        warm_up_steps,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.pad_token_idx = pad_token_idx
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.max_seq_len = max_seq_len
        self.schedule = schedule
        self.warm_up_steps = warm_up_steps
        self.dropout = dropout

        if self.schedule == "transformer":
            assert warm_up_steps is not None, "A value for warm_up_steps is required for transformer LR schedule"

        # Additional args passed in to **kwargs in init will also be saved
        self.save_hyperparameters()

        # These must be set by subclasses
        self.sampler = None
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, x):
        raise NotImplementedError()

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor)
        """

        raise NotImplementedError()

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        self.train()

        model_output = self.forward(batch)
        loss, token_mask_loss, attn_loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)

        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True)
        self.log("train_token", token_acc, on_step=True, logger=True, sync_dist=True)
        self.log("token_mask_loss", token_mask_loss, prog_bar=True, logger=False, sync_dist=True)
        self.log("attn_loss", attn_loss, prog_bar=True, logger=False, sync_dist=True)

        train_loss = {
            "loss" : loss,
            "token_mask_loss" : token_mask_loss,
            "attn_loss": attn_loss
        }
        return train_loss
    
    def training_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]
        atom_tokens_org = batch["atom_tokens_org"]

        loss, token_mask_loss, attn_loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        # perplexity = self._calc_perplexity(batch, model_output)
        # mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        # metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        # mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
        # invalid = torch.tensor(metrics["invalid"], device=loss.device)
        mol_strs, mol_tokens = self.get_str(model_output)

        total = len(target_smiles)
        acc_str = 0
        for i in range(len(target_smiles)):
            # print("smiles len: " + str(len(mol_strs[i])) + " /  " + str(len(target_smiles[i])))
            flag = True
            for j in range(len(atom_tokens_org[i])):
                if atom_tokens_org[i][j] != mol_tokens[i][j]:
                    flag = False
            if flag:
                acc_str += 1
            
            # if mol_strs[i] == target_smiles[i]:
            #     acc_str += 1
        acc_str = acc_str / total

        # Log for prog bar only
        # self.log("mol_acc", mol_acc, prog_bar=True, logger=False, sync_dist=True)
        self.log("acc_str", acc_str, prog_bar=True, logger=False, sync_dist=True)
        self.log("token_acc", token_acc, prog_bar=True, logger=False, sync_dist=True)

        val_outputs = {
            "val_loss": loss,
            "val_token_acc": token_acc,
            # "perplexity": perplexity,
            # "val_molecular_accuracy": mol_acc,
            # "val_invalid_smiles": invalid
        }
        return val_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def test_step(self, batch, batch_idx):
        self.eval()
        self.test_sampling_alg = "greedy"

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]
        source_smiles = batch["source_smiles"]
        atom_tokens_org = batch["atom_tokens_org"]

        loss, token_mask_loss, attn_loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        # perplexity = self._calc_perplexity(batch, model_output)
        # mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
        # metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)


        mol_strs, mol_tokens = self.get_str(model_output)

        with open("result.txt", "a") as f:
            for i in range(len(target_smiles)):
                f.write("pred: " + mol_strs[i] + "\n")
                f.write("targ: " + target_smiles[i] + "\n")
                f.write("sour: " + source_smiles[i] + "\n")
                flag = True
                for j in range(len(atom_tokens_org[i])):
                    if atom_tokens_org[i][j] != mol_tokens[i][j]:
                        flag = False
                
                # if mol_strs[i] == target_smiles[i]:
                #     f.write("True\n")
                # else:
                #     f.write("False\n")
                if flag:
                    f.write("True\n")
                else:
                    f.write("False\n")

        test_outputs = {
            "test_loss": loss.item(),
            "test_token_acc": token_acc,
            # "test_perplexity": perplexity,
            # "test_invalid_smiles": metrics["invalid"]
        }

        # if self.test_sampling_alg == "greedy":
        #     test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

        # elif self.test_sampling_alg == "beam":
        #     test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
        #     test_outputs["test_molecular_top_1_accuracy"] = metrics["top_1_accuracy"]
        #     test_outputs["test_molecular_top_2_accuracy"] = metrics["top_2_accuracy"]
        #     test_outputs["test_molecular_top_3_accuracy"] = metrics["top_3_accuracy"]
        #     test_outputs["test_molecular_top_5_accuracy"] = metrics["top_5_accuracy"]
        #     test_outputs["test_molecular_top_10_accuracy"] = metrics["top_10_accuracy"]

        # else:
        #     raise ValueError(f"Unknown test sampling algorithm, {self.test_sampling_alg}")

        return test_outputs

    def test_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        if self.schedule == "const":
            print("Using constant LR schedule.")
            const_sch = FuncLR(optim, lr_lambda=self._const_lr)
            sch = {"scheduler": const_sch, "interval": "step"}

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

    def _const_lr(self, step):
        if self.warm_up_steps is not None and step < self.warm_up_steps:
            return (self.lr / self.warm_up_steps) * step
        
        # return self.lr / (int(step / 79) * 40)
        return self.lr

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        """ 
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = - (1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_token_acc(self, batch_input, model_output):
        # token_ids = batch_input["target"]
        # target_mask = batch_input["target_mask"]
        token_ids = batch_input["atom_token_ids"]
        target_mask = batch_input["atom_tokens_pad_masks"]
        token_output = model_output["token_output"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def _avg_dicts(self, colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Pre-train Models --------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class BARTModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            pad_token_idx,
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers // 2, norm=enc_norm)
        assert d_model % num_heads == 0
        dim_head = d_model // num_heads
        attention = []
        for _ in range(3):
            attention.append(Attention(d_model, pos_emb=None, edge_dim=d_model, dim_head=dim_head, heads=num_heads))
        
        # self.graph_enc = GraphTransformer( 
        #     input_dim=9,
        #     h_dim=d_model,
        #     depth=3,
        #     attention=attention,
        #     edge_input_dim=9,
        #     edge_h_dim=d_model,
        #     # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
        #     with_feedforwards=True,
        #     # whether to add a feedforward after each attention layer, suggested by literature to be needed
        #     gated_residual=True,  # to use the gated residual to prevent over-smoothing
        #     rel_pos_emb=True
        # )

        self.graph_enc = GraphCrossformer(
            input_dim=9,
            h_dim=d_model,
            depth=3,
            corss_d_feedforward=d_feedforward,
            cross_dropout=dropout,
            cross_activation=activation,
            edge_input_dim=9,
            edge_h_dim=d_model,
            # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
            with_feedforwards=True,
            # whether to add a feedforward after each attention layer, suggested by literature to be needed
            gated_residual=True,  # to use the gated residual to prevent over-smoothing
            rel_pos_emb=True
        )
        
        # cross_norm = nn.LayerNorm(d_model)
        # cross_layer = PreNormCrossLayer(d_model, num_heads, d_feedforward, dropout, activation)
        # self.cross = PreNormCross(cross_layer, num_layers // 2, norm=cross_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        # self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)
        self.decoder = PreNormDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.loss_attn_fn = nn.MSELoss(reduction='sum')
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device)

        # memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        # graph
        atom = x["prods_atom"]
        edge = x["prods_edge"]
        length = x["lengths"]
        adj = x["prods_adj"]
        atom_masks = x["atom_masks"]
        
        # text_embs = self.text_enc(encoder_embs.transpose(0, 1), mask=encoder_pad_mask)
        text_embs = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        # node_embs, edge_embs = self.graph_enc(atom, edge, lengths=length, adj=adj)
        memory, att_weight, reorder_attn = self.graph_enc(atom, edge, lengths=length, adj=None, memory=text_embs,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=encoder_pad_mask.clone())
        
        # memory, att_weight = self.cross(
        #     # text_embs,
        #     # node_embs.transpose(1, 0),
        #     node_embs.transpose(1, 0),
        #     text_embs,
        #     tgt_mask=None,
        #     tgt_key_padding_mask=None,
        #     memory_key_padding_mask=encoder_pad_mask.clone()
        #     # memory_key_padding_mask=atom_masks.clone()

        # )

        # model_output, decoder_att_weight = self.decoder(
        #     decoder_embs,
        #     memory,
        #     tgt_mask=tgt_mask,
        #     tgt_key_padding_mask=decoder_pad_mask,
        #     memory_key_padding_mask=encoder_pad_mask.clone()
        # )

        # token_output = self.token_fc(model_output)
        token_output = self.token_fc(memory)

        output = {
            # "model_output": model_output,
            "token_output": token_output,
            # "decoder_att_weight": decoder_att_weight
            "reorder_attn": reorder_attn
        }

        return output

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        
        # graph
        atom = batch["prods_atom"]
        edge = batch["prods_edge"]
        length = batch["lengths"]
        adj = batch["prods_adj"]
        atom_masks = batch["atom_masks"]
        
        # model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        # text_embs = self.text_enc(encoder_embs.transpose(0, 1), mask=encoder_pad_mask)
        # node_embs, edge_embs = self.graph_enc(atom, edge, lengths=length, adj=adj)
        
        # model_output = self.cross_enc(
        #     text_embs,
        #     node=node_embs,
        #     mask=atom_masks.clone()
        # ).transpose(0, 1)

        text_embs = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        node_embs, edge_embs = self.graph_enc(atom, edge, lengths=length, adj=adj)
        
        model_output, att_weight = self.cross(
            text_embs,
            node_embs.transpose(1, 0),
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=atom_masks.clone()
        )
        
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embs.device)

        model_output, weight_attn = self.decoder(
            decoder_embs, 
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        # tokens = batch_input["target"]
        # pad_mask = batch_input["target_mask"]
        # target_smiles = batch_input["target_smiles"]

        # print("target_smiles", target_smiles)
        token_output = model_output["token_output"]
        pad_mask = batch_input["atom_tokens_pad_masks"]
        tokens = batch_input["atom_token_ids"]
        reorder_attn = model_output["reorder_attn"]
        atom_order = batch_input["atom_order"]
        # print("token_output", token_output.shape)
        # print("tokens", tokens.shape)
        # decoder_att_weight = model_output["decoder_att_weight"]
        # cross_attn = batch_input["cross_attn"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)
        # if cross_attn is not None:
        #     attns = decoder_att_weight[:, 1:, 1:-1]
        #     attns_masked = attns.masked_fill(~cross_attn.bool(),
        #                                            0.0)
        #     attn_loss = self.loss_attn_fn(attns_masked, cross_attn)
        #     # print("decoder_att_weight", decoder_att_weight.shape)
        #     # print("cross_attn", cross_attn.shape)
        
        if atom_order is not None:
            attns = reorder_attn
            attns_masked = attns.masked_fill(~atom_order.bool(),
                                                   0.0)
            attn_loss = self.loss_attn_fn(attns_masked, atom_order)
            # print("decoder_att_weight", decoder_att_weight.shape)
            # print("cross_attn", cross_attn.shape)

        loss = 1.0 * token_mask_loss + 1.0 * attn_loss
        # loss = token_mask_loss
        

        # attn_loss = torch.tensor(0.0, device=token_output.device)
        # loss = token_mask_loss

        # return token_mask_loss
        return loss, token_mask_loss, attn_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        # loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))
        loss = self.loss_fn(token_pred, target.reshape(-1))
        inv_target_mask = ~(target_mask > 0)
        inv_target_mask = inv_target_mask.reshape(-1)
        loss = loss * inv_target_mask
        loss = loss.sum()
        # inv_target_mask = ~(target_mask > 0)
        # num_tokens = inv_target_mask.sum()
        # loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input["encoder_input"]
        enc_mask = batch_input["encoder_pad_mask"]
        
        prods_atom = batch_input["prods_atom"]
        prods_adj = batch_input["prods_adj"]
        prods_edge = batch_input["prods_edge"]
        lengths = batch_input["lengths"]
        atom_masks = batch_input["atom_masks"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        encode_input = {
            "encoder_input": enc_input,
            "encoder_pad_mask": enc_mask,
            
            "prods_atom": prods_atom,
            "prods_adj": prods_adj,
            "prods_edge": prods_edge,
            "lengths": lengths,
            "atom_masks": atom_masks
        }
        memory = self.encode(encode_input)
        mem_mask = enc_mask.clone()

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, memory.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask
        }
        model_output = self.decode(decode_input)
        return model_output
    
    def get_str(self, model_output):
        token_output = model_output["token_output"]
        probs, output_ids = token_output.max(dim=2)

        tokens = output_ids.transpose(0, 1).tolist()
        tokens = self.sampler.tokeniser.convert_ids_to_tokens(tokens)
        mol_strs = self.sampler.tokeniser.detokenise(tokens)
        mol_tokens = self.sampler.tokeniser.detokens(tokens)

        return mol_strs, mol_tokens


class UnifiedModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            pad_token_idx,
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        enc_input = x["encoder_input"]
        enc_mask = x["encoder_pad_mask"]
        dec_input = x["decoder_input"]
        dec_mask = x["decoder_pad_mask"]
        att_mask = x["attention_mask"]

        model_input = torch.cat((enc_input, dec_input), dim=0)
        pad_mask = torch.cat((enc_mask, dec_mask), dim=0).transpose(0, 1)
        embs = self._construct_input(model_input)

        model_output = self.encoder(embs, mask=att_mask, src_key_padding_mask=pad_mask)
        token_output = self.token_fc(model_output)

        output = {
            "model_output": model_output,
            "token_output": token_output
        }

        return output

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        tgt_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, tgt_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size, _ = tuple(token_output.size())
        tgt_len, tgt_batch_size = tuple(target.size())

        assert seq_len == tgt_len
        assert batch_size == tgt_batch_size

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~target_mask
        num_tokens = inv_target_mask.sum()

        loss = loss * inv_target_mask
        loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_token_ids = batch_input["encoder_input"]
        enc_pad_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        enc_seq_len, batch_size = tuple(enc_token_ids.size())
        self.sampler.max_seq_len = self.max_seq_len - enc_seq_len

        decode_fn = partial(self._decode_fn, enc_token_ids=enc_token_ids, enc_pad_mask=enc_pad_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, enc_token_ids.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, enc_token_ids.device, k=self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, enc_token_ids, enc_pad_mask):
        # Strip off the start token for the decoded sequence
        dec_token_ids = token_ids[1:, :]

        enc_length, _ = tuple(enc_token_ids.shape)
        dec_length, _ = tuple(dec_token_ids.shape)
        att_mask = self._build_att_mask(enc_length - 1, dec_length + 1, device=dec_token_ids.device)

        model_input = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask,
            "decoder_input": dec_token_ids,
            "decoder_pad_mask": pad_mask[1:, :],
            "attention_mask": att_mask
        }
        token_output = self.forward(model_input)["token_output"]
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _build_att_mask(self, enc_length, dec_length, device="cpu"):
        seq_len = enc_length + dec_length
        enc_mask = torch.zeros((seq_len, enc_length), device=device)
        upper_dec_mask = torch.ones((enc_length, dec_length), device=device)
        lower_dec_mask = torch.ones((dec_length, dec_length), device=device).triu_(1)
        dec_mask = torch.cat((upper_dec_mask, lower_dec_mask), dim=0)
        mask = torch.cat((enc_mask, dec_mask), dim=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

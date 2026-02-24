# import torch
# from torch import nn
# import torch.nn.functional as F
# from caveat.models import Base

# class HouseholdDiscreteVAE(Base):
#     def __init__(self, encoding_weights=None, *args, **kwargs): # récupére les poids
#         # nombre de classe d'activité
#         self._encodings_backup = kwargs.get("encodings")
#         super().__init__(*args, **kwargs)

#         # permet de stocker les poids pour la loss
#         # et gérer le transfert GPU/CPU automatiquement
#         if encoding_weights is not None:
#             self.register_buffer('class_weights', encoding_weights)
#         else:
#             self.class_weights = None
#     def _embed_labels(self, labels):
#         '''
#         helper pour l'embedding des attributs socio démographique
#         chaque label a sa propre couche d'embedding
#         '''
#         embedded_labels = []
#         for i, layer in enumerate(self.label_embedding): # loop sur les couches d'embedding
#             embedded_labels.append(layer(labels[:, :, i].long())) # applique l'embedding
#         # pour chaque membre on a ses labels embeddés [B,5,E*len(self.label_cardinalities)]    
#         return torch.cat(embedded_labels, dim=-1)
#     def build(self,**config):
#         "architecture du modèle"
#         "config contient tous les 'model_params' du YAML et les paramètres de model()"

#         # paramètres du modèle
#         self.latent_dim_hh = config.get("latent_dim_hh", 32)
#         h_size = config.get("hidden_size", 64)
#         self.num_activites = self._encodings_backup or config.get("encodings")
#         self.num_activites = int(self.num_activites)

#         # géométrie
#         self.max_members = int(config.get('max_members', 5))
#         self.time_steps = int(config.get('time_steps', 144))

#         self.hh_size_col_idx = config.get("hh_size_col_idx", None) 
#         hh_size_map = config.get("hh_size_map", None)
#         if hh_size_map is not None:
#             self.register_buffer("hh_size_map_tensor", torch.tensor(hh_size_map, dtype=torch.long))
#         else:
#             self.hh_size_map_tensor = None
#         # paramètres de coordination
#         # n_coord_blocs = nombre de blocs de coordination 
#         # member_dilation = le membre i regarde à quelle portée ? ses voisins (1) ou ceux plus loin (2+)
#         self.n_coord_blocks = int(config.get("n_coord_blocks", 2))
#         self.member_dilation = int(config.get("member_dilation", 1))

#         # embedding des activités
#         self.embedding = nn.Embedding(self.num_activites, h_size)

#         # embedding des attributs
#         self.label_cardinalities = config.get("label_cardinalities", [])
#         self.label_embedding_dim = config.get("label_embedding_dim", 8)
#         self.label_embedding = nn.ModuleList([nn.Embedding(c, self.label_embedding_dim) for c in self.label_cardinalities]) # une couche par attribut

#         # taille des attributs
#         self.attr_per_member_dim = len(self.label_cardinalities) * self.label_embedding_dim

#         # slot embedding
#         # utilisé dans le decoder
#         self.slot_dim = config.get("slot_embedding_dim", 16)
#         self.slot_embedding = nn.Embedding(self.max_members, self.slot_dim)

        
#         # on n'inclut pas les dimensions du slot embedding car l'encodeur n'en a pas besoin
#         # et il pourrait apprendre via la position des membres plûtot que leurs caracs / la distribution
#         # on utilise les schedules + les attributs
#         self.input_channels = h_size + self.attr_per_member_dim

#         # encodeur individuel -> apprend les motifs temporels 
#         self.encoder_cnn = nn.Sequential(
#             nn.Conv2d(self.input_channels, h_size*2, kernel_size=(1, 3), stride=(1,2), padding = (0,1)),
#             nn.BatchNorm2d(h_size*2),
#             nn.ReLU(),

#             nn.Conv2d(h_size*2, h_size*4, kernel_size=(1,3), stride=(1,2), padding = (0,1)),
#             nn.BatchNorm2d(h_size*4),
#             nn.ReLU(),

#             nn.Conv2d(h_size*4, h_size*8, kernel_size=(1,3), stride=(1,2), padding = (0,1)),
#             nn.BatchNorm2d(h_size*8),
#             nn.ReLU(),
#         )
#         # Dimension après le premier CNN : Channels * Hauteur(5) * Largeur(18)
#         # on l'obtient dynamiquement
#         with torch.no_grad():
#             dummy = torch.zeros(1, self.input_channels, self.max_members, self.time_steps)
#             out = self.encoder_cnn(dummy)
#             self.c_out, self.h_out, self.w_out = out.shape[1], out.shape[2], out.shape[3]

#         # COORDINATION : mélange d'infos entre les membres
#         # plusieurs blocks de coordination en fonction de la config 
#         # un block = une convolution avec les autres membres + un SE sur tous les membres
#         self.enc_coord_blocks = nn.ModuleList([CoordinationBlock(self.c_out, dilation = self.member_dilation) for i in range(self.n_coord_blocks)])
        
#         # dimension de la grille avant compression pour le ménage
#         self.flat_per_member = self.c_out * 1 * self.w_out
#         # self.latent_input_dim = self.flat_per_member + self.attr_per_member_dim

#         # compression dans l'espace latent et non linéarité
#         self.fc_mu = nn.Sequential(
#             nn.Linear(self.c_out, self.c_out), nn.ReLU(),
#             nn.Linear(self.c_out, self.latent_dim_hh)
#         )
#         self.fc_var = nn.Sequential(
#             nn.Linear(self.c_out, self.c_out), nn.ReLU(),
#             nn.Linear(self.c_out, self.latent_dim_hh)
#         )
#         # on ajoute les attributs et l'embedding de slot avant la couche fc pour capturer des corrélations
#         # on recrée la grille
#         self.decoder_input = nn.Linear(self.latent_dim_hh + self.attr_per_member_dim + self.slot_dim, self.flat_per_member)
#         # self.unflatten = nn.Unflatten(1, (self.c_out, 1 , self.w_out))

#         # coordination dans le decoder
#         self.dec_coord_blocks = nn.ModuleList([
#             CoordinationBlock(self.c_out, dilation=self.member_dilation) 
#             for _ in range(self.n_coord_blocks)
#         ])

#         # le stride en largeur ajoute un stride-1 "0" entre chaque activité -> rajoute x2 pas de temps (le kernel ajoute les valeurs)
#         self.decoder_cnn = nn.Sequential(
#             nn.ConvTranspose2d(self.c_out, self.c_out//2, kernel_size=(1,3), stride = (1,2), padding=(0,1), output_padding=(0, 1)),
#             nn.BatchNorm2d(self.c_out//2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(self.c_out // 2, self.c_out// 4, kernel_size=(1,3), stride = (1,2),padding=(0,1),   output_padding=(0, 1)),
#             nn.BatchNorm2d(self.c_out//4),
#             nn.ReLU(),
#             # voir la formule de h_out mais globalement h_out = kernel
#             nn.ConvTranspose2d(self.c_out // 4, h_size, kernel_size=(1,3), stride = (1,2),padding=(0,1),  output_padding=(0, 1)),
#             nn.BatchNorm2d(h_size),
#             nn.ReLU(),
#             # couche final qui, contenu des 64 indices pour ce pas temps, quels sont les logits d'activité pour chaque individu
#             # c'est passer de 64 embeddings aux scores des 10 activités
#             nn.Conv2d(h_size, self.num_activites, kernel_size=1),

#             # passer des logits à des probabilités
#             nn.LogSoftmax(dim=1)

#         )

#     def forward(self, batch, **kwargs):
#         x = batch["x"] # [B,5,144]
#         labels = batch["labels"] # [B, 5, N_Labels]
#         mask = batch["mask"] # [B,5]
        
#         # taille originelle
#         B, M, T = x.shape

#         # on fait passer x dans la couche d'embedding
#         # c'est à dire chaque activité à chaque pas de temps
#         # en nombre, devient un indice pour la table d'embedding
#         # E[2] = [...,...,...]
#         # pour cela il faut des entiers (en indice) donc .long
#         # on permute pour retrouver la convention des images
#         x_emb = self.embedding(x.long()).permute(0,3,1,2)
#         embedded_labels = []
#         for i, layer in enumerate(self.label_embedding):
#             # on passe de [B,5,1] (le label encodé pour chaque individu)
#             # à [B,5,E] avec E l'embedding du label
#             # donc embedded_labels est une liste de tenseurs 
#             embedded_labels.append(layer(labels[:, :, i].long())) 
#         # pour chaque membre on a ses labels embeddés [B,5,E*len(self.label_cardinalities)]
#         labels_rep = torch.concat(embedded_labels, dim = -1)

#         # étend sur le temps les labels pour les coller aux activités
#         labels_map = labels_rep.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, T)

#         # concaténation
#         enc_in = torch.cat([x_emb, labels_map], dim=1)

#         # [B, C, 5, 18] post encoder
#         features = self.encoder_cnn(enc_in)  

#         for block in self.enc_coord_blocks:
#             features = block(features, mask) # on a toujours [B, C, 5, 18] 

#         # # on a pas capturé les interactions intra ménage
#         # # pour cela on utilise notre module d'attention
#         # # flatten channels*time pour le module
#         # features_flat = features.permute(0,2,1,3).reshape(B, M, -1)
#         # z_sched_context, attn_weights = self.hh_attention(features_flat, mask)
#         # # somme pondérée sur la dimension individu
#         # # Sum([B, M, Attr] * [B, M, 1]) -> [B, Attr]
#         # z_labels_context = torch.sum(labels_rep*attn_weights, dim = 1)
#         # # [B, Attr+labels = latent_input_dim]
#         # latent_space_input = torch.cat([z_sched_context, z_labels_context], dim = -1)

#         # pour le latent on doit compresser l'information des membres en une information ménage
#         # on utilise un pooling via moyenne en prenant en compte le mask pour ne pas biaiser la moyenne
#         mask_4d = mask.view(B, 1, M, 1) # pour matcher features
#         valid_pixels = (mask_4d.sum(dim = (2,3))*self.w_out).clamp(min = 1.0) # dénominateur
#         pooled = (features * mask_4d).sum(dim = (2,3)) / valid_pixels # moyenne pondérée
#         mu = self.fc_mu(pooled) # compression
#         logvar = self.fc_var(pooled) # compression
#         z = self.reparameterize(mu, logvar) # [B, latent]

#         # decoder_input_vec = torch.cat([z, z_labels_context], dim = -1)
#         log_probs = self._decode(z, labels_rep, mask) # passe dans le decoder

#         return [log_probs, mu, logvar, z, mask]
#     def _decode(self, z, labels_rep, mask):
#         M = self.max_members
#         B = z.shape[0]
#         # on applique l'embedding de slots
#         slot_ids = torch.arange(M, device=z.device).unsqueeze(0).expand(B,-1) # [B, M]
#         slots = self.slot_embedding(slot_ids) # [B, M, slots_dim]

#         # on copie le z pour chaque membre du ménage
#         z_expanded = z.unsqueeze(1).expand(B, M, -1)   # [B, M, latent]

#         # ajoute les attributs et les slots
#         dec_features = torch.cat([z_expanded, labels_rep, slots], dim = -1)  # [B, M, latent+slots_dim+labels_dim]
#         # ajoute la dimension temps compréssée
#         dec_features  = self.decoder_input(dec_features) # [B, M, self.c_out]
#         # bon format pour les images [B,C,M,T]
#         dec_features = dec_features.view(B, M, self.c_out, self.w_out).permute(0,2,1,3) # [B, self.c_out, M, T]
        
#         # coordination
#         for block in self.dec_coord_blocks:
#             dec_features = block(dec_features, mask)
        
#         log_probs = self.decoder_cnn(dec_features) # [B, Classes, 5, 144]

#         # 
#         log_probs = log_probs.permute(0,2,3,1)  # [B, 5, 144, Classes] les logits
#         return log_probs
    
#     def loss_function(self, args, batch, **kwargs):
#         log_probs = args[0]  # [B, 5, 144, Classes]
#         mu = args[1] # [B, 32]
#         logvar = args[2] # [B, 32]
#         z = args[3] # [B, 32]
#         mask = args[4] # [B, 5]

#         # target : [B,5,144]
#         x = batch["x"].long()

#         # loss de reconstruction (NLL)
#         # nécessite des probas de forme [B, Classes, 5, 144]
#         log_probs_for_loss = log_probs.permute(0,3,1,2)

#         # none reduction pour appliquer le mask après
#         weight = getattr(self, "class_weights", None)
#         recon_loss = F.nll_loss(
#                     log_probs_for_loss, 
#                     x, 
#                     weight=weight, 
#                     ignore_index=0, 
#                     reduction='none'
#                 )
#         # mettre les poids fréquence inverse
#         # mask = [B,5], on doit avoir [B,5,144] comme la target
#         # unsqueeze ajoute une dimension de 1, expand_as pour les matcher
#         mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss).float()    
        
#         # masquage et normalisation par pixel valide -> loss effective
#         denom = mask_expanded.sum().clamp(min=1.0)
#         recon_loss = (recon_loss * mask_expanded).sum() / denom

#         # kld par ménage
#         kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

#         # moyenne du batch
#         kld_loss = kld_per_sample.mean()
#         kld_weight = kwargs.get('kld_weight', 0.01)

#         # loss totale
#         total_loss = recon_loss + kld_weight*kld_loss

#         return {
#             "loss" : total_loss,
#             "recon_loss" : recon_loss,
#             "KLD" : kld_loss,
#             "z" : z 
#         }
    
#     def predict_step(self, batch, batch_idx, dataloader_idx = 0):

#         # en cas de test/validation
#         if isinstance(batch, dict):
#             x = batch["x"]
#             labels = batch["labels"]
            
#             # on fait passer dans le modèle complet (Encodeur -> Latent -> Décodeur)
#             outputs = self.forward(batch)
#             log_probs = outputs[0] 
#             z = outputs[3]         
            
#             return x, log_probs, z, labels

#         # en cas de génération
#         else:
#             labels, z = batch
#             B = z.shape[0]
#             M = self.max_members

            
            
#             # on récupère la vraie taille du ménage depuis les labels
#             if self.hh_size_col_idx is not None and self.hh_size_map_tensor is not None:
#                 # on prend le label du premier membre 
#                 # labels est [B, 5, N_Attr]
#                 encoded_sizes = labels[:, 0, self.hh_size_col_idx].long()
                
#                 # on traduit l'index (ex: 3) en nombre réel (ex: 3 personnes) grâce au mapping
#                 real_sizes = self.hh_size_map_tensor[encoded_sizes]
                
#                 # 2. On construit le masque [B, 5]
#                 # arange [0, 1, 2, 3, 4] vs real_sizes [1, 5, ...]
#                 # Si real_size = 1 -> [1, 0, 0, 0, 0]
#                 mask = (torch.arange(M, device=z.device).expand(B, -1) < real_sizes.unsqueeze(1)).float()
                
#             else:
#                 print("Warning: hh_size_col_idx non configuré, génération full members")
#                 mask = torch.ones(B, M, device=z.device)
#             # à changer pour matcher l'hétérogénéité de la distribution des ménages d'IDF
#             mask = torch.ones(z.shape[0], self.max_members, device=z.device)
            
#             # On génère via le décodeur uniquement
#             labels_rep = self._embed_labels(labels)
#             log_probs = self._decode(z, labels_rep, mask)

#             # --- 3. CONTRAINTES STRICTES (PAD vs NO-PAD) ---
            
#             # Préparation du masque pour qu'il ait la même forme que log_probs
#             # [B, 5] -> [B, 5, 1, 1] -> [B, 5, 144, Classes]
#             mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(log_probs).bool()

#             # A. INTERDICTION DU PAD POUR LES VIVANTS
#             # On prend les log_probs calculés par le modèle...
#             # ... mais on force l'index 0 (PAD) à -Infini pour qu'il ne soit jamais choisi.
#             active_log_probs = log_probs.clone()
#             active_log_probs[..., 0] = -1e9 

#             # B. OBLIGATION DU PAD POUR LES FANTÔMES
#             # On crée un tenseur qui dit "PAD à 100%"
#             inactive_log_probs = torch.full_like(log_probs, -1e9)
#             inactive_log_probs[..., 0] = 0.0 # Log(1) = 0

#             # C. FUSION INTELLIGENTE
#             # Si le masque est True (Vivant) -> On prend A (Interdiction PAD)
#             # Si le masque est False (Mort)  -> On prend B (Obligation PAD)
#             final_log_probs = torch.where(mask_expanded, active_log_probs, inactive_log_probs)
            
#             return labels, final_log_probs, z
#             return labels, log_probs, z
 
#     def training_step(self, batch, batch_idx):
#         # Forward Pass
#         outputs = self.forward(batch)
        
#         # calcul de la loss
#         # on récupère le poids KLD depuis les params de l'expérience (stockés dans self par Base)
#         kld_w = getattr(self, "kld_loss_weight", 0.01)
#         loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)
        
#         # logging
#         self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
#         self.log("train_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
#         self.log("train_kld", loss_dict["KLD"], on_step=False, on_epoch=True)
        
#         return loss_dict["loss"]

#     def validation_step(self, batch, batch_idx):
#         outputs = self.forward(batch)
#         kld_w = getattr(self, "kld_loss_weight", 0.01)
#         loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)
        
#         self.log("val_loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
#         self.log("val_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
#         return loss_dict["loss"]

#     def test_step(self, batch, batch_idx):
#         outputs = self.forward(batch)
#         kld_w = getattr(self, "kld_loss_weight", 0.01)
#         loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)
        
#         self.log("test_loss", loss_dict["loss"])
#         return loss_dict["loss"]
    
#     def on_validation_end(self):
#         pass
# # class HouseholdAttention(nn.Module):
# #     def __init__(self, input_dim, hidden_dim=64):
# #         super().__init__()
# #         # Calcule un score d'importance pour chaque membre
# #         self.attn_net = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.Tanh(),
# #             nn.Linear(hidden_dim, 1) 
# #         )

# #     def forward(self, x, mask):
# #         '''
# #         attribue des scores d'importance au schedule de chaque membre
# #         puis fait une somme pondérée par ces scores pour résumer le ménage en entier
# #         '''
# #          # [B, M, flat_dim]
# #         scores = self.attn_net(x) # [B, M, 1]
# #         scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9) # [B, M, 1]
# #         weights = F.softmax(scores, dim=1) # [B, M, 1]
# #         context = torch.sum(x * weights, dim=1) # [B, Flat_Dim]

# #         return context, weights

# class CoordinationBlock(nn.Module):
#     def __init__(self, channels, dilation = 1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # deux parties : le mix local via convolution et le mix global via fc
#         # convolution sur les membres
#         self.local_mix = nn.Conv2d(
#             channels, channels, kernel_size=(3,1), dilation=(dilation,1), padding=(dilation,0), bias = False
#         )

#         # inspiré de Squeeze and Excite
#         self.global_mix = nn.Sequential(
#             nn.Linear(channels, channels // 16),
#             nn.ReLU(),
#             nn.Linear(channels//16, channels),
#             nn.Sigmoid()
#         )

#         # normalisation par sample
#         # divise les channels par num_groups -> ça donne g groupes
#         # à l'intérieur du groupe BatchNorm
#         self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
#         self.relu = nn.ReLU()

#     def forward(self, x, mask):
#         B, C, M, W = x.shape
#         mask_4d = mask.view(B,1,M,1).float()

#         # assure le padding
#         x = x * mask_4d

#         # toujours [B,C,M,T]
#         out = self.local_mix(x)

#         # il faut d'abord faire la moyenne sur M et W avant le global mix
#         sum_pooled = (out * mask_4d).sum(dim=(2, 3))

#         # nombre de pixels valides
#         denom = (mask.sum(dim=1) * W).clamp(min=1.0).view(B, 1)

#         mean_pooled = sum_pooled / denom

#         # applique le mix global (active/désactive les canaux communs)
#         # on view pour avoir B, C, M, T (format image)
#         out_global = self.global_mix(mean_pooled).view(B,-1,1,1)

#         # applique la sigmoid à l'output
#         out = out * out_global

#         return self.relu(self.norm(out)) * mask_4d


import torch
from torch import nn
import torch.nn.functional as F
from caveat.models import Base

class HouseholdDiscreteVAE(Base):
    def __init__(self, encoding_weights=None, *args, **kwargs): 
        # récupère les poids pour la loss
        self._encodings_backup = kwargs.get("encodings")
        super().__init__(*args, **kwargs)

        # stockage des poids dans un buffer pour gestion automatique CPU/GPU
        if encoding_weights is not None:
            self.register_buffer('class_weights', encoding_weights)
        else:
            self.class_weights = None

    def _embed_labels(self, labels):
        '''
        Helper pour l'embedding des attributs socio-démographiques.
        Chaque attribut a sa propre couche d'embedding.
        '''
        embedded_labels = []
        for i, layer in enumerate(self.label_embedding): 
            embedded_labels.append(layer(labels[:, :, i].long()))
        # concaténation : [B, M, Somme_Dims_Embeddings]
        return torch.cat(embedded_labels, dim=-1)

    def _make_cnn(self, in_channels, architecture_list, mode='encoder'):
        """
        Construit dynamiquement une suite de couches CNN à partir d'une liste de configuration.
        """
        layers = []
        current_in = in_channels
        
        for layer_cfg in architecture_list:
            out_c = layer_cfg['out_channels']
            k = layer_cfg.get('kernel', (1, 3))
            s = layer_cfg.get('stride', (1, 2))
            p = layer_cfg.get('padding', (0, 1))
            
            if mode == 'encoder':
                layers.append(nn.Conv2d(current_in, out_c, kernel_size=k, stride=s, padding=p))
            else:
                op = layer_cfg.get('output_padding', (0, 0)) 
                layers.append(nn.ConvTranspose2d(current_in, out_c, kernel_size=k, stride=s, padding=p, output_padding=op))
            
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            current_in = out_c
            
        return nn.Sequential(*layers)

    def build(self, **config):
        """
        Construction dynamique de l'architecture du modèle basée sur la config YAML
        """
        self.latent_dim_hh = config.get("latent_dim_hh", 32)
        h_size = config.get("hidden_size", 64)

        # dimensions des données
        self.max_members = int(config.get('max_members', 5))
        self.time_steps = int(config.get('time_steps', 144))
        self.num_activites = int(self._encodings_backup or config.get("encodings"))

        # configuration du dropout
        self.use_dropout = config.get("use_dropout", False)
        self.dropout_rate = config.get("dropout_rate", 0.5)
        self.dropout_locations = config.get("dropout_locations", ["encoder", "latent", "decoder"])

        print(f"Dropout configuration: enabled={self.use_dropout}, rate={self.dropout_rate}, locations={self.dropout_locations}")

        self.hh_size_col_idx = config.get("hh_size_col_idx", None) 
        hh_size_map = config.get("hh_size_map", None)
        if hh_size_map is not None:
            self.register_buffer("hh_size_map_tensor", torch.tensor(hh_size_map, dtype=torch.long))
        else:
            self.hh_size_map_tensor = None

        # embedding des activités
        self.embedding = nn.Embedding(self.num_activites, h_size)

        # embedding des attributs
        self.label_cardinalities = config.get("label_cardinalities", [])
        self.label_embedding_dim = config.get("label_embedding_dim", 8)
        self.label_embedding = nn.ModuleList([
            nn.Embedding(c, self.label_embedding_dim) for c in self.label_cardinalities
        ])
        self.attr_per_member_dim = len(self.label_cardinalities) * self.label_embedding_dim

        # dropout sur l'embedding des activités (après concat)
        if self.use_dropout and "embedding" in self.dropout_locations:
            self.embedding_dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.embedding_dropout = nn.Identity()

        # entrée CNN = Embedding Activité + Embeddings Attributs
        self.input_channels = h_size + self.attr_per_member_dim


        default_enc = [
            {'out_channels': 64, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1)},
            {'out_channels': 128, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1)},
            {'out_channels': 256, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1)},
            {'out_channels': 512, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1)}
        ]
        enc_arch = config.get('encoder_architecture', default_enc)
        self.encoder_cnn = self._make_cnn(self.input_channels, enc_arch, mode='encoder')

        #  simule un passage pour éviter les calculs manuels de dimensions
        with torch.no_grad():
            # [Batch, Channels, Height(Members), Width(Time)]
            dummy_input = torch.zeros(1, self.input_channels, self.max_members, self.time_steps)
            dummy_out = self.encoder_cnn(dummy_input)
            
            self.c_out = dummy_out.shape[1] # channels en sortie du CNN
            self.h_out = dummy_out.shape[2] 
            self.w_out = dummy_out.shape[3] # temps compressé 
            print(f"Architecture CNN : Input {self.time_steps} -> Compressé à {self.w_out} (x{self.c_out} channels)")

        # coordination
        self.use_coordination = config.get("use_coordination", True)
        self.use_global_mix = config.get("use_global_mix", True)
        self.n_coord_blocks = int(config.get("n_coord_blocks", 2))
        self.member_dilation = int(config.get("member_dilation", 1))
        
        # on attend une liste dans le YAML, convertie en tuple
        self.coord_kernel_size = tuple(config.get("coord_kernel_size", [3, 1])) 
        self.global_mix_ratio = int(config.get("global_mix_ratio", 16))

        if self.use_coordination:
            self.enc_coord_blocks = nn.ModuleList([
                CoordinationBlock(
                    self.c_out, 
                    dilation=self.member_dilation, 
                    use_global_mix=self.use_global_mix,
                    kernel_size=self.coord_kernel_size,
                    global_mix_ratio=self.global_mix_ratio
                ) for _ in range(self.n_coord_blocks)
            ])
        else:
            self.enc_coord_blocks = nn.ModuleList([])

        # espace Latent
        if self.use_dropout and "latent" in self.dropout_locations:
            self.fc_mu = nn.Sequential(
                nn.Linear(self.c_out, self.c_out), nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.c_out, self.latent_dim_hh)
            )
            self.fc_var = nn.Sequential(
                nn.Linear(self.c_out, self.c_out), nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.c_out, self.latent_dim_hh)
            )
        else:
            self.fc_mu = nn.Sequential(
                nn.Linear(self.c_out, self.c_out), nn.ReLU(),
                nn.Linear(self.c_out, self.latent_dim_hh)
            )
            self.fc_var = nn.Sequential(
                nn.Linear(self.c_out, self.c_out), nn.ReLU(),
                nn.Linear(self.c_out, self.latent_dim_hh)
            )

        # taille de la grille aplatie pour le ménage entier
        self.flat_per_member = self.c_out * 1 * self.w_out

        # probleme initial : on passe de latent_dim_hh + attr_dim -> flat_per_member (énorme couche linéaire)
        # pour éviter de l'overfitting et instabilité, on utilise des couches progressives avec ou sans dropout
        input_dim = self.latent_dim_hh + self.attr_per_member_dim
        if self.use_dropout and "decoder" in self.dropout_locations:
            self.decoder_input = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(512, self.flat_per_member)
            )
        else:
            self.decoder_input = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, self.flat_per_member)
            )
# essayer avec que le premier individu par ménage
        # blocs de coordination décodeur 
        if self.use_coordination:
            self.dec_coord_blocks = nn.ModuleList([
                CoordinationBlock(
                    self.c_out, 
                    dilation=self.member_dilation, 
                    use_global_mix=self.use_global_mix,
                    kernel_size=self.coord_kernel_size,
                    global_mix_ratio=self.global_mix_ratio
                ) for _ in range(self.n_coord_blocks)
            ])
        else:
            self.dec_coord_blocks = nn.ModuleList([])

     
        default_dec = [
            {'out_channels': 256, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1), 'output_padding': (0,1)},
            {'out_channels': 128, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1), 'output_padding': (0,1)},
            {'out_channels': 64, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1), 'output_padding': (0,1)},
            {'out_channels': 64, 'kernel': (1,3), 'stride': (1,2), 'padding': (0,1), 'output_padding': (0,1)}
        ]
        dec_arch = config.get('decoder_architecture', default_dec)
        
        self.decoder_cnn = self._make_cnn(self.c_out, dec_arch, mode='decoder')
        
        # projection finale vers les logits d'activité
        last_conv_channels = dec_arch[-1]['out_channels']
        self.final_projection = nn.Sequential(
            nn.Conv2d(last_conv_channels, self.num_activites, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, batch, **kwargs):
        x = batch["x"] # [B, M, T]
        labels = batch["labels"] # [B, M, N_Labels]
        mask = batch["mask"] # [B, M]
        
        B, M, T = x.shape

        # embedding Activités
        x_emb = self.embedding(x.long()).permute(0,3,1,2) # [B, H_size, M, T]
        
        # embedding Attributs
        labels_rep = self._embed_labels(labels) # [B, M, Attr_dim]
        labels_map = labels_rep.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, T)

        # concaténation
        enc_in = torch.cat([x_emb, labels_map], dim=1)

        # dropout si configuré
        enc_in = self.embedding_dropout(enc_in)

        # encodage CNN
        features = self.encoder_cnn(enc_in) # [B, C_out, M, T_compressed]

        # coordination
        for block in self.enc_coord_blocks:
            features = block(features, mask) 

        # pooling (Moyenne pondérée par le masque) 
        mask_4d = mask.view(B, 1, M, 1)
        valid_pixels = (mask_4d.sum(dim=(2,3)) * self.w_out).clamp(min=1.0)
        pooled = (features * mask_4d).sum(dim=(2,3)) / valid_pixels
        
        # espace Latent
        mu = self.fc_mu(pooled)
        logvar = self.fc_var(pooled)
        z = self.reparameterize(mu, logvar) # [B, latent_dim]

        # décodage
        log_probs = self._decode(z, labels_rep, mask)

        return [log_probs, mu, logvar, z, mask]

    def _decode(self, z, labels_rep, mask):
        B = z.shape[0]
        M = self.max_members
        
        # on copie le z pour chaque membre du ménage : [B, M, latent]
        z_expanded = z.unsqueeze(1).expand(B, M, -1)
        
        # on concatène Latent + Attributs
        dec_input_vec = torch.cat([z_expanded, labels_rep], dim=-1)

        # projection linéaire pour retrouver la grille (with dropout if configured)
        if isinstance(self.decoder_input, nn.Sequential):
            dec_features = self.decoder_input(dec_input_vec)
        else:
            dec_features = self.decoder_input(dec_input_vec) # [B, M, flat_per_member]
        
        # reshape pour format image [B, Channels, M, T_compressed]
        dec_features = dec_features.view(B, M, self.c_out, self.w_out).permute(0,2,1,3)
        
        # coordination Décodeur 
        for block in self.dec_coord_blocks:
            dec_features = block(dec_features, mask)
        
        # CNN Décodeur 
        dec_features = self.decoder_cnn(dec_features)
        
        # projection finale (Logits)
        log_probs = self.final_projection(dec_features) # [B, Classes, M, T]
        
        return log_probs.permute(0,2,3,1) # [B, M, T, Classes]

    def loss_function(self, args, batch, **kwargs):
        log_probs = args[0]  # [B, 5, 144, Classes]
        mu = args[1] 
        logvar = args[2] 
        z = args[3] 
        mask = args[4] # [B, 5]

        x = batch["x"].long()

        
        log_probs_for_loss = log_probs.permute(0,3,1,2)
        # loss de reconstruction 
        # none reduction pour appliquer le mask après
        weight = getattr(self, "class_weights", None)
        recon_loss = F.nll_loss(
                    log_probs_for_loss, 
                    x, 
                    weight=weight, 
                    ignore_index=0, 
                    reduction='none'
                )
        # mask = [B,5], on doit avoir [B,5,144] comme la target
        mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss).float()    
        
        # masquage et normalisation par pixel valide -> loss effective
        denom = mask_expanded.sum().clamp(min=1.0)
        recon_loss = (recon_loss * mask_expanded).sum() / denom

        # kld PAR MENAGE
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # moyenne du batch
        kld_loss = kld_per_sample.mean()
        kld_weight = kwargs.get('kld_weight', 0.01)

        total_loss = recon_loss + kld_weight*kld_loss

        return {
            "loss" : total_loss,
            "recon_loss" : recon_loss,
            "KLD" : kld_loss,
            "z" : z 
        }

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        # CAS 1 : Validation / Test 
        if isinstance(batch, dict):
            # on exécute le modèle
            outputs = self.forward(batch)
            log_probs = outputs[0]
            z = outputs[3]
            
            return batch["x"], log_probs, z, batch["labels"]

        # CAS 2 : génération (Tuple : Z, labels)
        else:
            z, labels = batch
            B = z.shape[0]
            M = self.max_members

            # masquage dynamique
            if self.hh_size_col_idx is not None and self.hh_size_map_tensor is not None:
                encoded_sizes = labels[:, 0, self.hh_size_col_idx].long() 
                real_sizes = self.hh_size_map_tensor[encoded_sizes]
                # masque : 1 si présent, 0 si absent
                mask = (torch.arange(M, device=z.device).expand(B, -1) < real_sizes.unsqueeze(1)).float()
            else:
                # Fallback
                mask = torch.ones(B, M, device=z.device)

            # décodage
            labels_rep = self._embed_labels(labels)

            # mask pour la coordination
            log_probs = self._decode(z, labels_rep, mask)

            # masque de la grille -> chaque pas de temps et pour chaque classe
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(log_probs).bool()

            # pour les réels : interdiction de générer PAD
            active_log_probs = log_probs.clone()
            active_log_probs[..., 0] = -1e9 

            # obligation du PAD pour les fantômes
            inactive_log_probs = torch.full_like(log_probs, -1e9)
            inactive_log_probs[..., 0] = 0.0 

            # fusion
            final_log_probs = torch.where(mask_expanded, active_log_probs, inactive_log_probs)
            
            # pour la génération, runners.py attend 3 valeurs : Labels, Output, Z
            return labels, final_log_probs, z

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        kld_w = getattr(self, "kld_loss_weight", 0.01)
        loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)
        self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
        self.log("train_kld", loss_dict["KLD"], on_step=False, on_epoch=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        kld_w = getattr(self, "kld_loss_weight", 0.01)
        loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)
        self.log("val_loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        kld_w = getattr(self, "kld_loss_weight", 0.01)
        loss_dict = self.loss_function(outputs, batch, kld_weight=kld_w)
        self.log("test_loss", loss_dict["loss"])
        return loss_dict["loss"]
    def on_validation_end(self):
        # On surcharge la méthode de Experiment pour ne rien faire.
        pass

class CoordinationBlock(nn.Module):
    def __init__(self, channels, dilation=1, use_global_mix=True, kernel_size=(3,1), global_mix_ratio=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_global_mix = use_global_mix
        
        # mix local
        # calcul automatique du padding pour garder la taille constante : p = (k-1) * d / 2
        # on suppose que le kernel est impair (3, 5, 7) sur la dimension hauteur (membres)
        k_h, k_w = kernel_size
        pad_h = (k_h - 1) * dilation // 2
        pad_w = (k_w - 1) // 2
        
        self.local_mix = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            dilation=(dilation, 1), 
            padding=(pad_h, pad_w), 
            bias=False
        )

        # mix global
        if self.use_global_mix:
            # sécurité 
            reduced_channels = max(1, channels // global_mix_ratio)
            
            self.global_mix = nn.Sequential(
                nn.Linear(channels, reduced_channels),
                nn.ReLU(),
                nn.Linear(reduced_channels, channels),
                nn.Sigmoid()
            )
        
        # normalisation
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        B, C, M, W = x.shape
        mask_4d = mask.view(B,1,M,1).float()

        # on sauvegarde l'entrée pour la connexion résiduelle (skip connection)
        residual = x

        # assure que les membres vides restent à 0
        x = x * mask_4d

        # mélange local : convolutions sur chaque membre indépendamment
        out = self.local_mix(x)

        # mélange global : agrège l'information de tous les membres pour moduler les features
        if self.use_global_mix:
            # moyenne sur Membres et Temps
            sum_pooled = (out * mask_4d).sum(dim=(2, 3))
            denom = (mask.sum(dim=1) * W).clamp(min=1.0).view(B, 1)
            mean_pooled = sum_pooled / denom

            # calcul des poids d'attention globaux
            out_global = self.global_mix(mean_pooled).view(B, -1, 1, 1)

            # aspplication
            out = out * out_global

        # connection résiduelle pour faire passer le gradient
        out = out + residual

        # Résidu + Norm + ReLU + Re-Masquage
        return self.relu(self.norm(out)) * mask_4d
import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, factors=8, layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factors = factors
        self.layers = layers
        
        # GMF part
        self.embedding_user_gmf = nn.Embedding(num_users, factors)
        self.embedding_item_gmf = nn.Embedding(num_items, factors)
        
        # MLP part
        self.embedding_user_mlp = nn.Embedding(num_users, layers[0] // 2)
        self.embedding_item_mlp = nn.Embedding(num_items, layers[0] // 2)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        for idx in range(1, len(layers)):
            self.fc_layers.append(nn.Linear(layers[idx-1], layers[idx]))
        
        # Final prediction layer
        self.affine_output = nn.Linear(layers[-1] + factors, 1)
        self.logistic = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, user, item):
        # GMF part
        user_gmf = self.embedding_user_gmf(user)
        item_gmf = self.embedding_item_gmf(item)
        gmf_vector = user_gmf * item_gmf
        
        # MLP part
        user_mlp = self.embedding_user_mlp(user)
        item_mlp = self.embedding_item_mlp(item)
        vector = torch.cat((user_mlp, item_mlp), dim=-1)
        for idx in range(len(self.layers) - 1):
            vector = self.fc_layers[idx](vector)
            vector = nn.ReLU()(vector)
        
        # Concatenate GMF and MLP parts
        predict_vector = torch.cat((gmf_vector, vector), dim=-1)
        
        # Final prediction
        logits = self.affine_output(predict_vector)
        rating = self.logistic(logits)
        return rating

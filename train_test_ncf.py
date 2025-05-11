import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, preprocess_data
from dataset import MovieLensDataset
from ncf_model import NCF


def train_model(epochs=100):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, item, rating in train_loader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            optimizer.zero_grad()
            prediction = model(user, item).view(-1)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss : {total_loss:.4f}")  # earlier  / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print("\n Evaluating ...")
            evaluate_model(top_k=10)
            print("\n")


def evaluate_model(top_k=10):
    model.eval()
    hr = []
    user_ratings = {}  # Store test ratings per user
    with torch.no_grad():
        # Group test ratings by user
        for user, item, rating in test_loader:
            user, item = user.cpu().numpy(), item.cpu().numpy()
            rating = rating.cpu().numpy()
            for u, i, r in zip(user, item, rating):
                if u not in user_ratings:
                    user_ratings[u] = []
                user_ratings[u].append((i, r))
    
    # Evaluate HR@K for each user
    with torch.no_grad():
        for user in user_ratings:
            # Predict scores for all movies
            all_items = torch.arange(model.num_items, dtype=torch.long, device=device)
            user_tensor = torch.full_like(all_items, user, dtype=torch.long)
            predictions = model(user_tensor, all_items).view(-1)
            pred_scores = predictions.cpu().numpy()
            
            # Get top-K predicted items
            top_k_indices = np.argsort(pred_scores)[-top_k:][::-1]  # Top-K indices in descending order
            
            # Get relevant items (rated 1 in test set)
            relevant_items = set(i for i, r in user_ratings[user] if r == 1)
            
            # Compute HR@K: 1 if any top-K item is relevant, 0 otherwise
            hits = len(set(top_k_indices) & relevant_items)
            hr.append(1.0 if hits > 0 else 0.0)
    
    global best_hit_ratio
    hit_ratio = np.mean(hr) if hr else 0.0
    print(f"HR@{top_k}: {hit_ratio:.4f}")
    if hit_ratio > best_hit_ratio:
        print(f"Improved hit ratio, saving this model ...")
        torch.save(model.state_dict(), 'best_ncf_model.pth')
        best_hit_ratio = hit_ratio


if __name__ == "__main__":
    torch.manual_seed(23)
    np.random.seed(23)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    file_path = 'ml-1m/' 
    ratings, movies, users = load_data(file_path)
    train_data, test_data, num_users, num_items = preprocess_data(ratings)
    
    train_dataset = MovieLensDataset(train_data)
    test_dataset = MovieLensDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    best_hit_ratio = 0

    model = NCF(num_users, num_items, factors=8, layers=[64, 32, 16, 8])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(epochs=1000)
    evaluate_model(top_k=10)

import numpy as np
import torch
import pandas as pd
from ncf_model_for_mab import NCF


def heuristic_reward(user_id, movie_id):
    user_tensor = torch.tensor([user_to_idx[user_id]], dtype=torch.long, device=device)
    movie_tensor = torch.tensor([movie_to_idx[movie_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        predicted_rating = model(user_tensor, movie_tensor).item()
    # Add small noise to simulate uncertainty
    noise = np.random.normal(0, 0.01)
    return predicted_rating + noise


def optimal_reward(user_id):
    user_tensor = torch.tensor([user_to_idx[user_id]], dtype=torch.long, device=device)
    all_movies = torch.arange(len(movie_ids), dtype=torch.long, device=device)
    user_tensor = user_tensor.repeat(len(movie_ids))
    with torch.no_grad():
        predictions = model(user_tensor, all_movies).view(-1)
    return predictions.max().item()


def run_linucb_simulation(T=1000, alpha=1.0, lambda_=0.1, print_interval=100):
    # Initialize LinUCB parameters
    num_movies = len(movie_ids)
    d = user_embeddings.shape[1]  # Dimension of concatenated embedding (8 + 32 = 40)
    A = [lambda_ * np.eye(d) for _ in range(num_movies)]
    b = [np.zeros(d) for _ in range(num_movies)]

    # Track metrics
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    rewards = []

    # Simulate T time steps
    for t in range(T):
        # Sample a random user from training data
        user_id = np.random.choice(user_ids)
        user_idx = user_to_idx[user_id]
        # Get concatenated user embedding
        user_embedding = user_embeddings[user_idx].cpu().numpy()  # Shape: (40,)
        x_t = user_embedding.flatten()

        # Compute UCB for each movie
        ucb_scores = []
        for i in range(num_movies):
            A_inv = np.linalg.inv(A[i])
            theta_i = A_inv @ b[i]
            ucb = x_t @ theta_i + alpha * np.sqrt(x_t @ A_inv @ x_t)
            ucb_scores.append(ucb)

        # Choose movie with highest UCB
        chosen_movie_idx = np.argmax(ucb_scores)
        chosen_movie_id = movie_ids[chosen_movie_idx]

        # Get heuristic reward
        r_t = heuristic_reward(user_id, chosen_movie_id)

        # Compute optimal reward for regret
        r_opt = optimal_reward(user_id)
        regret_t = r_opt - r_t

        # Update metrics
        cumulative_reward += r_t
        cumulative_regret += regret_t
        rewards.append(r_t)

        # Update LinUCB parameters
        A[chosen_movie_idx] += np.outer(x_t, x_t)
        b[chosen_movie_idx] += r_t * x_t

        # Print metrics at intervals
        if (t + 1) % print_interval == 0:
            avg_reward = cumulative_reward / (t + 1)
            avg_regret = cumulative_regret / (t + 1)
            print(f"Step {t + 1}/{T}:")
            print(f"  Cumulative Reward: {cumulative_reward:.4f}")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Cumulative Regret: {cumulative_regret:.4f}")
            print(f"  Average Regret: {avg_regret:.4f}")

    # Final summary
    avg_reward = cumulative_reward / T
    avg_regret = cumulative_regret / T
    print("\nFinal Summary:")
    print(f"  Total Steps: {T}")
    print(f"  Final Cumulative Reward: {cumulative_reward:.4f}")
    print(f"  Final Average Reward: {avg_reward:.4f}")
    print(f"  Final Cumulative Regret: {cumulative_regret:.4f}")
    print(f"  Final Average Regret: {avg_regret:.4f}")


# Run the simulation
if __name__ == "__main__":
    torch.manual_seed(23)
    np.random.seed(23)

    file_path = 'ml-1m/'
    ratings = pd.read_csv(file_path + 'ratings.dat', sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'])
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_to_idx = {userId: idx for idx, userId in enumerate(user_ids)}
    movie_to_idx = {movieId: idx for idx, movieId in enumerate(movie_ids)}

    model = NCF(len(user_ids), len(movie_ids))
    model.load_state_dict(torch.load('best_ncf_model.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    gmf_user_embeddings = model.embedding_user_gmf.weight.data
    mlp_user_embeddings = model.embedding_user_mlp.weight.data
    user_embeddings = torch.cat((gmf_user_embeddings, mlp_user_embeddings), dim=1)  # Shape: (num_users, 8 + 32)
    average_user_embedding = torch.mean(user_embeddings, dim=0).cpu().numpy()  # Shape: (40,)

    run_linucb_simulation(T=1000, alpha=1.0, lambda_=0.1, print_interval=100)

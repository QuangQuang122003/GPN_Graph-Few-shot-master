import torch
import torch.nn.functional as F
from utils import load_data, task_generator, euclidean_dist, accuracy, f1
from models import GPN_Encoder, GPN_Valuator
import networkx as nx
import matplotlib.pyplot as plt

def load_model(model_path, input_dim, hidden_dim, dropout):
    encoder = GPN_Encoder(nfeat=input_dim, nhid=hidden_dim, dropout=dropout)
    scorer = GPN_Valuator(nfeat=input_dim, nhid=hidden_dim, dropout=dropout)
    checkpoint = torch.load(model_path)
    
    print("Checkpoint encoder weight shape:", checkpoint['encoder_state_dict']['gc1.weight'].shape)
    print("Model encoder weight shape:", encoder.gc1.weight.shape)
    
    if checkpoint['encoder_state_dict']['gc1.weight'].shape != encoder.gc1.weight.shape:
        print("Adjusting weight shapes to match the current model...")
        checkpoint['encoder_state_dict']['gc1.weight'] = checkpoint['encoder_state_dict']['gc1.weight'][:encoder.gc1.weight.shape[0], :]

    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    scorer.load_state_dict(checkpoint['valuator_state_dict'], strict=False)
    encoder.eval()
    scorer.eval()
    return encoder, scorer

def predict(encoder, scorer, features, adj, degrees, support_nodes, query_nodes, n_way, k_shot, cuda):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    support_embeddings = embeddings[support_nodes]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[query_nodes]

    support_degrees = torch.log(degrees[support_nodes].view([n_way, k_shot]))
    support_scores = scores[support_nodes].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    predicted_labels = torch.argmax(output, dim=1)
    return predicted_labels

def plot_predictions(predicted_labels, n_way):
    labels, counts = torch.unique(predicted_labels, return_counts=True)
    labels = labels.numpy()
    counts = counts.numpy()

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, tick_label=[f'Class {i}' for i in range(n_way)])
    plt.xlabel('Classes')
    plt.ylabel('Number of nodes')
    plt.title('Predicted Labels Distribution')
    plt.show()

def visualize_data(adj, features):
    G = nx.from_numpy_array(adj)

    plt.figure(figsize=(10, 10))
    nx.draw(G, node_size=10, node_color='blue', edge_color='gray')
    plt.title('Network Graph Visualization')
    plt.show()

    if features.shape[1] > 1:
        plt.scatter(features[:, 0], features[:, 1], c='blue', s=1)
        plt.title('Node Features Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

def plot_predictions_on_graph(G, predicted_labels):
    pos = nx.spring_layout(G)  
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=10, node_color=predicted_labels, cmap=plt.cm.jet, edge_color='gray')
    plt.title('Predicted Labels Visualization on Network Graph')
    plt.show()

if __name__ == '__main__':
    dataset = 'Amazon_clothing'
    model_path = 'gpn_model.pth'
    n_way = 5
    k_shot = 5
    n_query = 20
    cuda = torch.cuda.is_available()

    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset)

    print("Feature shape:", features.shape)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        degrees = degrees.cuda()

    # Load model đã huấn luyện
    input_dim = features.shape[1]
    hidden_dim = 16  
    dropout = 0.5 
    encoder, scorer = load_model(model_path, input_dim, hidden_dim, dropout)

    id_support, id_query, class_selected = task_generator(id_by_class, class_list_test, n_way, k_shot, n_query)

    # Dự đoán lớp cho các nút truy vấn
    predicted_labels = predict(encoder, scorer, features, adj, degrees, id_support, id_query, n_way, k_shot, cuda)
    print(f'Predicted Labels: {predicted_labels}')

    # Trực quan hóa kết quả dự đoán
    plot_predictions(predicted_labels, n_way)

 

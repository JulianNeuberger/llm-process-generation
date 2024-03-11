import tqdm
from matplotlib import pyplot as plt

from data import pet
import networkx

documents = pet.NewPetFormatImporter("../res/data/pet/all.new.jsonl").do_import()
document: pet.PetDocument
for document in tqdm.tqdm(documents):
    graph = networkx.DiGraph()
    labels = {}
    for i, mention in enumerate(document.mentions):
        if mention.type.lower() not in [
            "activity",
            "xor gateway",
            "and gateway",
            "condition specification",
        ]:
            continue
        graph.add_node(i)
        labels[i] = mention.text(document)
    for i, relation in enumerate(document.relations):
        if relation.type.lower() != "flow":
            continue
        graph.add_edge(relation.head_mention_index, relation.tail_mention_index)
    fig = plt.figure(figsize=(9.2, 13.6))
    pos = networkx.fruchterman_reingold_layout(graph)
    networkx.draw_networkx(graph, labels=labels, with_labels=True, pos=pos)
    plt.savefig(f"../figures/graphs/{document.id}.png")
    plt.close(fig)

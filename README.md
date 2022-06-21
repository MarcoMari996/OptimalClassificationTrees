# OptimalClassificationTrees
Self-made (not perfect) implementation of Optimal Classification Trees by Bertsimas and Dunn (Mach Learn (2017) 106:1039–1082)<br>
<a href="https://www.mit.edu/~dbertsim/papers/Machine%20Learning%20under%20a%20Modern%20Optimization%20Lens/Optimal_classification_trees_MachineLearning.pdf">Link to the Article</a>

Il progetto affronta il problema della classificazione di un insieme di dati da un punto di vista diverso da quello tipico del machine learning: l'obbiettivo è, infatti, quello di definire la creazione di alberi decisionali attraverso la risoluzione di un problema di ottimizzazione lineare vincolata, anziché attraverso algoritmi di generazione di branch empirici, come per esempio ID3. 

Il problema principale di tali metodi è la loro natura “top-down”: gli splits vengono determinati a partire dal nodo radice fino alle foglie considerando, tra tutti gli split possibili, quello che minimizza l’errore di classificazione, ma, una volta deciso lo split, la decisione viene iterata senza considerare l’impatto dei precedenti splits o di quelli futuri. 

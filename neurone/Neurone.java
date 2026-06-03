public abstract class Neurone implements iNeurone {
	protected double[] poids;
	protected double biais;
	protected double valeurSortie;
	protected double tauxApprentissage = 0.001;

	public Neurone(int nbEntrees) {
		poids = new double[nbEntrees];
		// Initialisation aléatoire des poids
		for (int i = 0; i < nbEntrees; i++) {
			poids[i] = Math.random() * 2 - 1;
		}
		biais = Math.random() * 2 - 1;
	}

	@Override
	public void metAJour(double[] entrees) {
		double sommeInterne = biais;
		for (int i = 0; i < entrees.length; i++) {
			sommeInterne += entrees[i] * poids[i];
		}
		this.valeurSortie = activation(sommeInterne);
	}

	@Override
	public double sortie() {
		return this.valeurSortie;
	}

	@Override
	public void apprentissage(double[][] entrees, double[] consigne, double MSElimite) {
		double mse = Double.MAX_VALUE;
		int maxEpochs = 10000;
		int epoch = 0;

		while (mse > MSElimite && epoch < maxEpochs) {
			double erreurTotale = 0;

			for (int i = 0; i < entrees.length; i++) {
				metAJour(entrees[i]);
				double erreur = consigne[i] - this.valeurSortie;

				// Mise à jour des poids et du biais
				for (int j = 0; j < poids.length; j++) {
					poids[j] += tauxApprentissage * erreur * entrees[i][j];
				}
				biais += tauxApprentissage * erreur;

				erreurTotale += erreur * erreur;
			}

			mse = erreurTotale / entrees.length;
			epoch++;
		}
	}

	// Méthode abstraite que chaque type de neurone devra définir
	protected abstract double activation(double sommeInterne);
}
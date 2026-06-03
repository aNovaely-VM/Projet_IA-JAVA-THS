public interface iNeurone {
	void metAJour(double[] entrees);

	double sortie();

	void apprentissage(double[][] entrees, double[] consigne, double MSElimite);
}
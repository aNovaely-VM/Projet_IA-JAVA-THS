public class NeuroneHeavyside extends Neurone {

	public NeuroneHeavyside(int nbEntrees) {
		super(nbEntrees);
	}

	@Override
	protected double activation(double sommeInterne) {
		// Fonction Heaviside : retourne 1 si somme >= 0, sinon 0
		return sommeInterne >= 0 ? 1.0 : 0.0;
	}
}
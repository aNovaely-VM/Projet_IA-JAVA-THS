public class NeuroneSigmoide extends Neurone
{
	// Fonction d'activation d'un neurone (peut facilement être modifiée par héritage)
        @Override
	protected float activation(final float valeur) {return 1.f / (1.f + (float) Math.exp(-valeur));}
	
	// Constructeur
	public NeuroneSigmoide(final int nbEntrees) {super(nbEntrees);}
}

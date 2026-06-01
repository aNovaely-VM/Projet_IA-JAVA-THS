public class NeuroneReLu extends Neurone
{
	// Fonction d'activation d'un neurone (peut facilement être modifiée par héritage)
	protected float activation(final float valeur) {return valeur >= 0 ? valeur : 0.f;}
	
	// Constructeur
	public NeuroneReLu(final int nbEntrees) {super(nbEntrees);}
}

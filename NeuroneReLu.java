// NeuroneReLU.java
public class NeuroneReLu extends Neurone {

    // Constructeur
    public NeuroneReLu(final int nbEntrees) {
        super(nbEntrees);
    }

    // Utilise la fonction max(0.0f, valeur) [cite: 6, 10, 99]
    @Override
    protected float activation(final float valeur) {
        return Math.max(0.0f, valeur);
    }
}
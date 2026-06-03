// NeuroneSigmoide.java
public class NeuroneSigmoide extends Neurone {

    // Constructeur
    public NeuroneSigmoide(final int nbEntrees) {
        super(nbEntrees);
    }

    // Utilise la fonction logistique continue 1/(1 + e^(-valeur)) [cite: 5, 10,
    // 106]
    @Override
    protected float activation(final float valeur) {
        return (float) (1.0 / (1.0 + Math.exp(-valeur)));
    }
}
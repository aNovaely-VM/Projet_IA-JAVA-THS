public class NeuroneSigmoide extends Neurone {

    public NeuroneSigmoide(int nbEntrees) {
        super(nbEntrees);
    }

    @Override
    protected double activation(double sommeInterne) {
        // Formule de la sigmoïde
        return 1.0 / (1.0 + Math.exp(-sommeInterne));
    }
}
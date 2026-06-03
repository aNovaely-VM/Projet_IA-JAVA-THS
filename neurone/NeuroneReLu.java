public class NeuroneReLU extends Neurone {

    public NeuroneReLU(int nbEntrees) {
        super(nbEntrees);
    }

    @Override
    protected double activation(double sommeInterne) {
        // Formule mathématique : max(0, x)
        return Math.max(0.0, sommeInterne);
    }
}

// ClassifieurComparatif.java
import java.util.List;

public class ClassifieurComparatif {
    public static void main(String[] args) {
        String dossierTrain = "dataset_groupe_6/train";
        String dossierTest = "dataset_groupe_6/test";

        System.out.println("Préparation de l'environnement comparatif...");
        // CORRECTION 1 : Remplacement de Image.Mode par Image.ModeCouleur
        // NOTE : On active le miroir (true) pour le train et (false) pour le test pour
        // rester cohérent
        List<Image> trainImages = ClassifieurChienChat.chargerDataset(dossierTrain, Image.ModeCouleur.GRIS, true);
        List<Image> testImages = ClassifieurChienChat.chargerDataset(dossierTest, Image.ModeCouleur.GRIS, false);

        if (trainImages.isEmpty()) {
            System.out.println("Dataset vide.");
            return;
        }

        ClassifieurChienChat.melangerFisherYates(trainImages);
        int nbPixels = trainImages.get(0).donnees().length;

        // CORRECTION 2 : Correction orthographique de NeuroneReLU (LU en majuscules)
        Neurone[] modeles = {
                new NeuroneHeavyside(nbPixels),
                new NeuroneSigmoide(nbPixels),
                new NeuroneReLu(nbPixels)
        };

        // CORRECTION 3 : Remplacement de .repeat() par des chaînes fixes pour
        // contourner le problème Java 8
        String ligneDouble = "=========================================================================================================";
        String ligneSimple = "---------------------------------------------------------------------------------------------------------";

        System.out.println("\n" + ligneDouble);
        System.out.printf("%-18s | %-12s | %-10s | %-15s | %-10s | %-10s | %-10s\n",
                "Type de Neurone", "MSE Finale", "Epoques", "Taux Réussite", "Précision", "Rappel", "Score F1");
        System.out.println(ligneSimple);

        for (Neurone modele : modeles) {
            String nom = modele.getClass().getSimpleName();

            // Entraînement
            int epoques = ClassifieurChienChat.entrainerPlafonne(modele, trainImages);
            float finalMse = calculerMse(modele, trainImages); // Fonction utilitaire locale

            // Évaluation silencieuse qui retourne [accuracy, precision, recall, f1]
            float[] metrics = evaluerSilencieux(modele, testImages);

            System.out.printf("%-18s | %-12.4f | %-10d | %-14.2f%% | %-10.2f | %-10.2f | %-10.2f\n",
                    nom, finalMse, epoques, metrics[0] * 100, metrics[1], metrics[2], metrics[3]);
        }
        System.out.println(ligneDouble);
    }

    private static float calculerMse(Neurone n, List<Image> train) {
        float mse = 0;
        for (Image img : train) {
            // CORRECTION 4 : Conversion obligatoire de int[] vers float[] pour la méthode
            // metAJour()
            int[] donnéesBrutes = img.donnees();
            float[] entreesTest = new float[donnéesBrutes.length];
            for (int k = 0; k < donnéesBrutes.length; k++) {
                entreesTest[k] = (float) donnéesBrutes[k];
            }

            n.metAJour(entreesTest);

            // CORRECTION 5 : Alignement de la cible (1.0f pour un chat) avec le reste du
            // projet
            float cible = (img.label() == 1) ? 1.0f : 0.0f;
            float delta = cible - n.sortie();
            mse += delta * delta;
        }
        return mse / train.size();
    }

    private static float[] evaluerSilencieux(Neurone n, List<Image> test) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (Image img : test) {
            // CORRECTION 6 : Même conversion obligatoire de int[] vers float[] pour
            // l'évaluation
            int[] donnéesBrutes = img.donnees();
            float[] entreesTest = new float[donnéesBrutes.length];
            for (int k = 0; k < donnéesBrutes.length; k++) {
                entreesTest[k] = (float) donnéesBrutes[k];
            }

            n.metAJour(entreesTest);
            boolean predictChat = n.sortie() >= 0.5f;
            boolean estChat = img.label() == 1;

            if (predictChat && estChat)
                tp++;
            else if (!predictChat && !estChat)
                tn++;
            else if (predictChat && !estChat)
                fp++;
            else
                fn++;
        }
        float accuracy = (float) (tp + tn) / test.size();
        float precision = (tp + fp) == 0 ? 0 : (float) tp / (tp + fp);
        float recall = (tp + fn) == 0 ? 0 : (float) tp / (tp + fn);
        float f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
        return new float[] { accuracy, precision, recall, f1 };
    }
}
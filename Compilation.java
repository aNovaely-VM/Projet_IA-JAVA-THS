import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Point d'entrée du programme de classification d'images par réseau de
 * neurones.
 * Modes : train, test, sort. Architectures : 1 neurone (binaire cat/non-cat) ou
 * 3 neurones (cat/dog/wild).
 * Voir README.md pour les instructions.
 */
public class Compilation {

    // =========================================================
    // CONFIGURATION
    // =========================================================

    static final String DATASET_TRAIN = "dataset_groupe_6/train/";
    static final String DATASET_TEST = "dataset_groupe_6/test/";
    static final String DATASET_SORT = "dataset_groupe_6/naming/to_sort/";
    static final String DATASET_SORT_OUT_BINAIRE = "dataset_groupe_6/naming/binaire/";
    static final String DATASET_SORT_OUT_MULTIPLE = "dataset_groupe_6/naming/multiple/";

    // Index parallèles : CLASSES[k] correspond à LABELS[k]
    static final String[] CLASSES = { "cat", "dog", "wild" };
    static final int[] LABELS = { 0, 1, 2 };

    static final int RESIZE = 64;

    private static final String[] OPTIONS_COULEUR = { "GRI", "COL", "TSL" };
    private static final String[] OPTIONS_NEURONE = { "heavyside", "relu", "sigmoide" };
    private static final String[] OPTIONS_TRAITEMENT = { "simple", "mirror", "histogramme (Oblige COL ou TSL)",
            "HOG (Oblige GRI)" };
    private static final String[] OPTIONS_ARCHITECTURE = { "1 Neurone (Cat ou non - Binaire)",
            "3 Neurones (Cat, Dog, Wild - Multiclasse)" };
    private static final String[] OPTIONS_ORDRE = { "Ordre séquentiel (Linéaire)", "Mélange aléatoire (Shuffle)",
            "Mélange à chaque itération (Deep Shuffle ─ très coûteux)" };
    private static final String[] OPTIONS_TYPE_LOGS = { "Binaire (1 Neurone)", "Multiple (3 Neurones)" };
    private static final String[] OPTIONS_OUI_NON = { "Oui", "Non" };

    // Paramètres actifs (sélectionnés par l'utilisateur au runtime)
    static String typeCouleurActif = "GRI";
    static String typeNeuroneActif = "sigmoide";
    static String typeTraitementActif = "simple";
    static boolean modeMonoNeuroneActif = false;
    static boolean modeShuffleActif = false;
    static boolean modeDeepShuffleActif = false;
    static boolean modeNormalisationActif = false;

    // Hyperparamètres
    static final float MSE_LIMITE = 0.01f;
    static final float ETA = 0.0001f;
    static final int MAX_ITERATIONS = 200;

    static final String SAVE_DIR_BINAIRE = "Logs/binaire/";
    static final String SAVE_DIR_MULTIPLE = "Logs/multiple/";

    // Cache des vecteurs transformés : clé = chemin absolu du fichier image.
    // Évite de recalculer HOG/miroir/etc. pour une même image entre train et test.
    private static final Map<String, float[]> cacheTransformations = new HashMap<>();

    // =========================================================
    // HELPERS
    // =========================================================

    static String saveDir() {
        return modeMonoNeuroneActif ? SAVE_DIR_BINAIRE : SAVE_DIR_MULTIPLE;
    }

    static String sortOutDir() {
        return modeMonoNeuroneActif ? DATASET_SORT_OUT_BINAIRE : DATASET_SORT_OUT_MULTIPLE;
    }

    // =========================================================
    // MAIN
    // =========================================================

    public static void main(String[] args) throws Exception {
        System.out.println("┌──────────────────────────────────────────┐");
        System.out.println("│  Compilation.java ─ Réseau de neurones   │");
        System.out.println("└──────────────────────────────────────────┘\n");

        Scanner sc = new Scanner(System.in);

        while (true) {
            System.out.println("Que voulez-vous faire ?");
            System.out.println("  1) Entraîner puis évaluer (Train + Test)");
            System.out.println("  2) Entraîner puis trier (Train + Sort)");
            System.out.println("  3) Entraîner uniquement (Train only)");
            System.out.println("  4) Évaluer uniquement (Test only ─ ! Nécessite un modèle déjà entraîné ! )");
            System.out.println("  5) Trier uniquement (Sort only ─ ! Nécessite un modèle déjà entraîné ! )");
            System.out.println(
                    "  6) Nettoyer les sorties (Clear ─ supprime les images triées dans naming/binaire et naming/multiple)");
            System.out.println("  7) Quitter le programme");
            System.out.print("\nVotre choix [1-7] : ");
            String choix = sc.nextLine().trim();
            System.out.println();

            if (choix.equals("7")) {
                System.out.println("Au revoir !");
                break;
            }

            boolean necessiteEntrainement = choix.equals("1") || choix.equals("2") || choix.equals("3");

            if (necessiteEntrainement) {
                typeNeuroneActif = menuChoix(sc, "Choisissez le type de neurone :", OPTIONS_NEURONE);

                String choixTraitementRaw = menuChoix(sc, "Choisissez le type de traitement :", OPTIONS_TRAITEMENT);
                // On extrait le premier mot pour ignorer les parenthèses d'aide à l'affichage
                typeTraitementActif = choixTraitementRaw.contains(" ")
                        ? choixTraitementRaw.split(" ")[0].trim()
                        : choixTraitementRaw.trim();

                // HOG et Histogramme imposent leur mode couleur
                if (typeTraitementActif.equalsIgnoreCase("HOG")) {
                    System.out.println(
                            "[INFO] Le mode HOG requiert des niveaux de gris. Configuration automatique sur 'GRI'.\n");
                    typeCouleurActif = "GRI";
                } else if (typeTraitementActif.equalsIgnoreCase("histogramme")) {
                    System.out.println(
                            "[INFO] Le mode Histogramme requiert de la couleur. Configuration automatique sur 'COL'.\n");
                    typeCouleurActif = "COL";
                } else {
                    typeCouleurActif = menuChoix(sc, "Choisissez le mode couleur :", OPTIONS_COULEUR);
                }

                String choixArch = menuChoix(sc, "Choisissez l'architecture du réseau :", OPTIONS_ARCHITECTURE);
                modeMonoNeuroneActif = choixArch.startsWith("1");

                String choixOrdre = menuChoix(sc, "Choisissez l'ordre des images :", OPTIONS_ORDRE);
                modeDeepShuffleActif = choixOrdre.contains("Deep Shuffle");
                modeShuffleActif = choixOrdre.contains("Shuffle") && !modeDeepShuffleActif;

                String choixNorm = menuChoix(sc, "Activer la normalisation des vecteurs d'entrée ?", OPTIONS_OUI_NON);
                modeNormalisationActif = choixNorm.equals("Oui");

                // Invalidation du cache : la nouvelle configuration peut produire des vecteurs
                // différents
                cacheTransformations.clear();
                afficherConfig();

            } else if (choix.equals("4") || choix.equals("5")) {
                String choixLogs = menuChoix(sc, "Quel type de modèle voulez-vous utiliser ?", OPTIONS_TYPE_LOGS);
                modeMonoNeuroneActif = choixLogs.startsWith("Binaire");

                File configFile = new File(saveDir() + "config.txt");
                if (!configFile.exists()) {
                    System.err.println("╔══════════════════════════════════════════════════════════╗");
                    System.err.println("║  ERREUR : Aucune configuration trouvée dans " + saveDir());
                    System.err.println("║  Impossible de lancer cette action sans modèle entraîné.");
                    System.err.println("║  → Lancez d'abord un entraînement (options 1, 2 ou 3).  ");
                    System.err.println("╚══════════════════════════════════════════════════════════╝");
                    continue;
                }

                try {
                    chargerConfiguration();
                    System.out.println("[INFO] Logs de configuration chargés depuis : " + saveDir());
                } catch (IOException e) {
                    System.err.println("[ERREUR] Lecture du fichier de configuration échouée : " + e.getMessage());
                    System.err.println("→ Relancez un entraînement pour régénérer la configuration.");
                    continue;
                }
                afficherConfig();
            }

            switch (choix) {
                case "1" -> {
                    train();
                    test();
                }
                case "2" -> {
                    train();
                    sort();
                }
                case "3" -> train();
                case "4" -> test();
                case "5" -> sort();
                case "6" -> clear(sc);
                default -> System.out.println("Choix invalide.\n");
            }

            System.out.println();
        }
    }

    private static String menuChoix(Scanner sc, String titre, String[] options) {
        while (true) {
            System.out.println("- " + titre);
            for (int i = 0; i < options.length; i++) {
                System.out.printf("  %d) %s\n", (i + 1), options[i]);
            }
            System.out.print("Votre choix : ");
            try {
                int choix = Integer.parseInt(sc.nextLine().trim());
                if (choix >= 1 && choix <= options.length) {
                    System.out.println();
                    return options[choix - 1];
                }
            } catch (NumberFormatException e) {
                /* saisie non numérique → reboucle */ }
            System.out.println("Choix invalide. Réessayez.\n");
        }
    }

    static void afficherConfig() {
        System.out.println("┌────────────────────────────────────────────────────────┐");
        System.out.println("│                 CONFIGURATION ACTIVE                   │");
        System.out.println("├────────────────────────────────────────────────────────┤");
        System.out.printf("│  Dataset train    : %-34s │\n", DATASET_TRAIN);
        System.out.printf("│  Dataset test     : %-34s │\n", DATASET_TEST);
        System.out.printf("│  Dataset à trier  : %-34s │\n", DATASET_SORT);
        System.out.printf("│  Sortie du tri    : %-34s │\n", sortOutDir());
        System.out.printf("│  Dossier logs     : %-34s │\n", saveDir());
        System.out.printf("│  Architecture     : %-34s │\n",
                (modeMonoNeuroneActif ? "1 Neurone (Cat / Non-Cat)" : "3 Neurones (Collaboratif)"));
        System.out.printf("│  Neurone choisi   : %-34s │\n", typeNeuroneActif);
        System.out.printf("│  Mode image       : %-34s │\n", typeCouleurActif);
        System.out.printf("│  Traitement       : %-34s │\n", typeTraitementActif);
        System.out.printf("│  Dimension d'adm. : %-34s │\n", (RESIZE + "x" + RESIZE));
        System.out.printf("│  MSE cible        : %-34s │\n", MSE_LIMITE);
        System.out.printf("│  Max Itérations   : %-34s │\n", MAX_ITERATIONS);
        System.out.printf("│  Pas d'app. (Eta) : %-34s │\n", ETA);
        System.out.printf("│  Normalisation    : %-34s │\n", (modeNormalisationActif ? "Activée" : "Désactivée"));
        System.out.printf("│  Deep Shuffle     : %-34s │\n",
                (modeDeepShuffleActif ? "Activé (shuffle/itération)" : "Désactivé"));
        System.out.println("└────────────────────────────────────────────────────────┘\n");
    }

    // =========================================================
    // PERSISTANCE DE LA CONFIGURATION
    // =========================================================

    /**
     * Sauvegarde les paramètres actifs dans config.txt pour pouvoir recharger le
     * bon mode lors d'un test/tri ultérieur.
     */
    private static void sauvegarderConfiguration() throws IOException {
        File dir = new File(saveDir());
        if (!dir.exists())
            dir.mkdirs();

        try (PrintWriter pw = new PrintWriter(new FileWriter(saveDir() + "config.txt"))) {
            pw.println("typeCouleurActif=" + typeCouleurActif);
            pw.println("typeNeuroneActif=" + typeNeuroneActif);
            pw.println("typeTraitementActif=" + typeTraitementActif);
            pw.println("modeMonoNeuroneActif=" + modeMonoNeuroneActif);
            pw.println("modeShuffleActif=" + modeShuffleActif);
            pw.println("modeDeepShuffleActif=" + modeDeepShuffleActif);
            pw.println("modeNormalisationActif=" + modeNormalisationActif);
        }
    }

    /**
     * Relit config.txt et réhydrate les variables statiques. Split limité à 2 parts
     * pour tolérer un "=" dans les valeurs.
     */
    private static void chargerConfiguration() throws IOException {
        File configFile = new File(saveDir() + "config.txt");
        if (!configFile.exists())
            throw new FileNotFoundException("Fichier de configuration absent : " + saveDir() + "config.txt");

        try (BufferedReader br = new BufferedReader(new FileReader(configFile))) {
            String ligne;
            while ((ligne = br.readLine()) != null) {
                String[] parts = ligne.split("=", 2);
                if (parts.length == 2) {
                    String cle = parts[0].trim(), valeur = parts[1].trim();
                    switch (cle) {
                        case "typeCouleurActif" -> typeCouleurActif = valeur;
                        case "typeNeuroneActif" -> typeNeuroneActif = valeur;
                        case "typeTraitementActif" -> typeTraitementActif = valeur;
                        case "modeMonoNeuroneActif" -> modeMonoNeuroneActif = Boolean.parseBoolean(valeur);
                        case "modeShuffleActif" -> modeShuffleActif = Boolean.parseBoolean(valeur);
                        case "modeDeepShuffleActif" -> modeDeepShuffleActif = Boolean.parseBoolean(valeur);
                        case "modeNormalisationActif" -> modeNormalisationActif = Boolean.parseBoolean(valeur);
                    }
                }
            }
        }
    }

    // =========================================================
    // VECTORISATION DES IMAGES
    // =========================================================

    static float[][] chargerEntrees(List<Image> images) {
        if (images.isEmpty())
            return new float[0][0];
        float[][] entrees = new float[images.size()][];
        for (int i = 0; i < images.size(); i++)
            entrees[i] = obtenirVecteurTransforme(images.get(i));
        return entrees;
    }

    /**
     * Retourne le vecteur transformé depuis le cache, ou le calcule et le mémorise.
     */
    static float[] obtenirVecteurTransforme(Image img) {
        float[] enCache = cacheTransformations.get(img.chemin());
        if (enCache != null)
            return enCache;
        float[] transforme = appliquerTransformations(img);
        cacheTransformations.put(img.chemin(), transforme);
        return transforme;
    }

    /**
     * Applique la chaîne complète : conversion colorimétrique → traitement
     * optionnel → normalisation.
     */
    static float[] appliquerTransformations(Image img) {
        int[] pixelsBruts = img.donnees();
        float[] donneesModifiees;
        int largeur = img.largeur(), hauteur = img.hauteur();

        switch (typeCouleurActif) {
            case "GRI" -> {
                // Luminance pondérée Rec.601 : meilleure perception visuelle que la moyenne
                // simple
                donneesModifiees = new float[pixelsBruts.length];
                for (int i = 0; i < pixelsBruts.length; i++) {
                    if (pixelsBruts[i] > 255 || pixelsBruts[i] < 0) {
                        int r = (pixelsBruts[i] >> 16) & 0xFF;
                        int g = (pixelsBruts[i] >> 8) & 0xFF;
                        int b = pixelsBruts[i] & 0xFF;
                        donneesModifiees[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.f;
                    } else {
                        donneesModifiees[i] = pixelsBruts[i] / 255.f;
                    }
                }
            }
            case "TSL" -> donneesModifiees = convertRGBtoTSL(pixelsBruts, img.estEnNiveauxDeGris());
            default -> {
                // Mode COL : R, G, B sont placés consécutivement → vecteur 3× plus long qu'en
                // GRI
                donneesModifiees = new float[pixelsBruts.length * (img.estEnNiveauxDeGris() ? 1 : 3)];
                if (img.estEnNiveauxDeGris()) {
                    for (int i = 0; i < pixelsBruts.length; i++)
                        donneesModifiees[i] = pixelsBruts[i] / 255.f;
                } else {
                    int index = 0;
                    for (int p : pixelsBruts) {
                        donneesModifiees[index++] = ((p >> 16) & 0xFF) / 255.f;
                        donneesModifiees[index++] = ((p >> 8) & 0xFF) / 255.f;
                        donneesModifiees[index++] = (p & 0xFF) / 255.f;
                    }
                }
            }
        }

        boolean estMulticomposante = typeCouleurActif.equals("COL") || typeCouleurActif.equals("TSL");

        switch (typeTraitementActif.toLowerCase()) {
            case "mirror" ->
                donneesModifiees = appliquerEffetMiroir(donneesModifiees, largeur, hauteur, estMulticomposante);
            case "histogramme" ->
                donneesModifiees = appliquerEgalisationHistogramme(donneesModifiees, estMulticomposante);
            case "hog" -> donneesModifiees = calculerHOG(donneesModifiees, largeur, hauteur, estMulticomposante);
        }

        if (modeNormalisationActif)
            donneesModifiees = normaliserVecteur(donneesModifiees);
        return donneesModifiees;
    }

    /**
     * Convertit des pixels RGB packed en vecteur TSL (Teinte, Saturation,
     * Luminosité). 3 composantes par pixel.
     */
    private static float[] convertRGBtoTSL(int[] rbgColor, boolean estGris) {
        float[] tsl = new float[rbgColor.length * 3];

        if (estGris) {
            for (int i = 0; i < rbgColor.length; i++) {
                tsl[3 * i] = 0.0f; // Teinte indéfinie
                tsl[3 * i + 1] = 0.0f; // Saturation nulle
                tsl[3 * i + 2] = rbgColor[i] / 255.f;
            }
            return tsl;
        }

        for (int i = 0; i < rbgColor.length; i++) {
            int p = rbgColor[i];
            float r = ((p >> 16) & 0xFF) / 255.f;
            float g = ((p >> 8) & 0xFF) / 255.f;
            float b = (p & 0xFF) / 255.f;

            float max = Math.max(r, Math.max(g, b));
            float min = Math.min(r, Math.min(g, b));
            float l = (max + min) / 2.0f;
            float h = 0.0f, s = 0.0f;

            if (max != min) {
                float d = max - min;
                s = l > 0.5f ? d / (2.0f - max - min) : d / (max + min);

                if (max == r)
                    h = (g - b) / d + (g < b ? 6.0f : 0.0f);
                else if (max == g)
                    h = (b - r) / d + 2.0f;
                else
                    h = (r - g) / d + 4.0f;
                h /= 6.0f; // normalisation dans [0, 1]
            }

            tsl[3 * i] = h;
            tsl[3 * i + 1] = s;
            tsl[3 * i + 2] = l;
        }
        return tsl;
    }

    /**
     * Retourne une copie de l'image avec chaque ligne de pixels retournée
     * horizontalement (augmentation de données).
     */
    private static float[] appliquerEffetMiroir(float[] src, int large, int haut, boolean multicomposante) {
        float[] dest = new float[src.length];
        int pas = multicomposante ? 3 : 1;

        for (int y = 0; y < haut; y++) {
            for (int x = 0; x < large; x++) {
                int idxSrc = (y * large + x) * pas;
                int idxDest = (y * large + (large - 1 - x)) * pas;
                for (int c = 0; c < pas; c++)
                    dest[idxDest + c] = src[idxSrc + c];
            }
        }
        return dest;
    }

    /**
     * Égalisation d'histogramme via CDF pour améliorer le contraste.
     * En GRI : redistribution uniforme des intensités.
     * En COL/TSL : seule la luminosité est égalisée pour préserver les teintes.
     */
    private static float[] appliquerEgalisationHistogramme(float[] src, boolean multicomposante) {
        float[] dest = new float[src.length];
        int bins = 256;
        int[] histogramme = new int[bins];

        if (!multicomposante) {
            for (float val : src)
                histogramme[Math.min(bins - 1, Math.max(0, (int) (val * 255)))]++;

            int[] cdf = new int[bins];
            cdf[0] = histogramme[0];
            for (int i = 1; i < bins; i++)
                cdf[i] = cdf[i - 1] + histogramme[i];

            int cdfMin = 0;
            for (int i = 0; i < bins; i++) {
                if (cdf[i] > 0) {
                    cdfMin = cdf[i];
                    break;
                }
            }

            for (int i = 0; i < src.length; i++) {
                int bin = Math.min(bins - 1, Math.max(0, (int) (src[i] * 255)));
                dest[i] = (float) (cdf[bin] - cdfMin) / (src.length - cdfMin);
            }

        } else {
            System.arraycopy(src, 0, dest, 0, src.length);
            int totalPixels = src.length / 3;
            boolean estTSL = typeCouleurActif.equals("TSL");

            for (int i = 0; i < totalPixels; i++) {
                float v = estTSL ? src[3 * i + 2] : (src[3 * i] + src[3 * i + 1] + src[3 * i + 2]) / 3.0f;
                histogramme[Math.min(bins - 1, Math.max(0, (int) (v * 255)))]++;
            }

            int[] cdf = new int[bins];
            cdf[0] = histogramme[0];
            for (int i = 1; i < bins; i++)
                cdf[i] = cdf[i - 1] + histogramme[i];

            int cdfMin = 0;
            for (int i = 0; i < bins; i++) {
                if (cdf[i] > 0) {
                    cdfMin = cdf[i];
                    break;
                }
            }

            for (int i = 0; i < totalPixels; i++) {
                float vOriginale = estTSL ? src[3 * i + 2] : (src[3 * i] + src[3 * i + 1] + src[3 * i + 2]) / 3.0f;
                int bin = Math.min(bins - 1, Math.max(0, (int) (vOriginale * 255)));
                float vEgalisee = (float) (cdf[bin] - cdfMin) / (totalPixels - cdfMin);

                if (estTSL) {
                    dest[3 * i + 2] = vEgalisee;
                } else {
                    // Facteur proportionnel pour conserver les ratios RGB ; +0.001 évite la
                    // division par zéro
                    float facteur = vEgalisee / (vOriginale + 0.001f);
                    dest[3 * i] = Math.min(1.0f, src[3 * i] * facteur);
                    dest[3 * i + 1] = Math.min(1.0f, src[3 * i + 1] * facteur);
                    dest[3 * i + 2] = Math.min(1.0f, src[3 * i + 2] * facteur);
                }
            }
        }
        return dest;
    }

    /**
     * Calcule le descripteur HOG (Histogram of Oriented Gradients).
     * Cellules 8×8 px, 9 bins angulaires sur [0°, 180°] (gradients non orientés).
     * Sortie : vecteur de taille (largeur/8) × (hauteur/8) × 9.
     */
    private static float[] calculerHOG(float[] src, int large, int haut, boolean multicomposante) {
        // Conversion en carte d'intensité (luminance)
        float[][] intensite = new float[haut][large];
        int pas = multicomposante ? 3 : 1;
        for (int y = 0; y < haut; y++) {
            for (int x = 0; x < large; x++) {
                int idx = (y * large + x) * pas;
                intensite[y][x] = multicomposante
                        ? 0.299f * src[idx] + 0.587f * src[idx + 1] + 0.114f * src[idx + 2]
                        : src[idx];
            }
        }

        // Gradients par différences finies (les bords restent à 0, pas de pixel voisin)
        float[][] magnitude = new float[haut][large];
        float[][] orientation = new float[haut][large];
        for (int y = 1; y < haut - 1; y++) {
            for (int x = 1; x < large - 1; x++) {
                float gx = intensite[y][x + 1] - intensite[y][x - 1];
                float gy = intensite[y + 1][x] - intensite[y - 1][x];
                magnitude[y][x] = (float) Math.sqrt(gx * gx + gy * gy);
                float angle = (float) Math.toDegrees(Math.atan2(gy, gx));
                if (angle < 0)
                    angle += 180.0f;
                orientation[y][x] = angle;
            }
        }

        // Accumulation des magnitudes dans les histogrammes de cellules
        int tailleCellule = 8, numBins = 9;
        int cellulesX = large / tailleCellule, cellulesY = haut / tailleCellule;
        float[] vecteurHOG = new float[cellulesX * cellulesY * numBins];

        int indexVecteur = 0;
        for (int cy = 0; cy < cellulesY; cy++) {
            for (int cx = 0; cx < cellulesX; cx++) {
                float[] histCellule = new float[numBins];
                for (int y = 0; y < tailleCellule; y++) {
                    for (int x = 0; x < tailleCellule; x++) {
                        float mag = magnitude[cy * tailleCellule + y][cx * tailleCellule + x];
                        float ang = orientation[cy * tailleCellule + y][cx * tailleCellule + x];
                        int bin = (int) (ang / (180.0f / numBins));
                        if (bin >= numBins)
                            bin = numBins - 1;
                        histCellule[bin] += mag;
                    }
                }
                for (float v : histCellule)
                    vecteurHOG[indexVecteur++] = v;
            }
        }
        return vecteurHOG;
    }

    /**
     * Normalise un vecteur à norme L2 unitaire. Retourne le vecteur inchangé si la
     * norme est quasi-nulle.
     */
    private static float[] normaliserVecteur(float[] v) {
        double norme = 0.0;
        for (float val : v)
            norme += val * val;
        norme = Math.sqrt(norme);
        if (norme < 1e-8)
            return v;
        float[] normalise = new float[v.length];
        for (int i = 0; i < v.length; i++)
            normalise[i] = (float) (v[i] / norme);
        return normalise;
    }

    /**
     * Charge toutes les images .jpg/.png depuis les sous-dossiers de classes (cat/,
     * dog/, wild/).
     */
    static List<Image> chargerImages(String baseDir) {
        List<Image> liste = new ArrayList<>();
        boolean estGris = typeCouleurActif.equals("GRI");

        for (int c = 0; c < CLASSES.length; c++) {
            String dossier = baseDir + CLASSES[c] + "/";
            List<String> fichiers = Image.listeFichiers(dossier);
            if (fichiers == null || fichiers.isEmpty()) {
                System.out.printf("  Dossier vide ou introuvable : %s\n", dossier);
                continue;
            }
            int count = 0;
            for (String chemin : fichiers) {
                String low = chemin.toLowerCase();
                if (!low.endsWith(".jpg") && !low.endsWith(".png"))
                    continue;
                liste.add(new Image(chemin, LABELS[c], estGris));
                count++;
            }
            System.out.printf("  Vectorisation | %-5s : %d fichiers chargés\n", CLASSES[c], count);
        }
        return liste;
    }

    static Neurone creerNeurone(int nbEntrees) {
        return switch (typeNeuroneActif.toLowerCase()) {
            case "heavyside" -> new NeuroneHeavyside(nbEntrees);
            case "relu" -> new NeuroneReLu(nbEntrees);
            default -> new NeuroneSigmoide(nbEntrees);
        };
    }

    // =========================================================
    // ENTRAÎNEMENT
    // =========================================================

    /**
     * Entraîne 1 ou 3 neurones par descente de gradient (règle du perceptron,
     * one-vs-all).
     * S'arrête quand MSE < MSE_LIMITE ou après MAX_ITERATIONS passages complets sur
     * le dataset.
     */
    static void train() throws Exception {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" PHASE D'ENTRAÎNEMENT");
        System.out.println("════════════════════════════════════════════════════════");

        List<Image> images = chargerImages(DATASET_TRAIN);
        if (images.isEmpty()) {
            System.err.println(" Annulation : aucune donnée d'entraînement.");
            return;
        }

        if (modeDeepShuffleActif) {
            System.out.println("  Mode ordonnancement : Deep Shuffle (mélange à chaque itération)");
            Collections.shuffle(images);
        } else if (modeShuffleActif) {
            System.out.println("  Mode ordonnancement : Mélange aléatoire (Shuffle)");
            Collections.shuffle(images);
        } else {
            System.out.println("  Mode ordonnancement : Séquentiel linéaire");
        }

        float[][] entrees = chargerEntrees(images);
        int nbEntrees = entrees[0].length;
        System.out.printf("  Caractéristiques par neurone : %d entrées\n\n", nbEntrees);

        Neurone.fixeCoefApprentissage(ETA);

        int nbNeuronesAEntrainer = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] coucheNeurones = new Neurone[nbNeuronesAEntrainer];
        for (int k = 0; k < nbNeuronesAEntrainer; k++)
            coucheNeurones[k] = creerNeurone(nbEntrees);

        System.out.printf(" Lancement de l'optimisation (%d neurone(s) en compétition)...\n", nbNeuronesAEntrainer);

        long tempsDebutTotal = System.currentTimeMillis();
        double globaleMse = 1.0;
        int iter = 0;

        while (globaleMse > MSE_LIMITE && iter < MAX_ITERATIONS) {
            long tempsDebutIter = System.nanoTime();
            double sommeErreursCarrees = 0.0;

            if (modeDeepShuffleActif) {
                // Réordonnancement des images et de leurs vecteurs précalculés ensemble pour ne
                // pas désynchroniser le cache
                List<Integer> indices = new ArrayList<>();
                for (int i = 0; i < images.size(); i++)
                    indices.add(i);
                Collections.shuffle(indices);
                List<Image> imagesMelangees = new ArrayList<>(images.size());
                float[][] entreesMelangees = new float[images.size()][];
                for (int i = 0; i < indices.size(); i++) {
                    imagesMelangees.add(images.get(indices.get(i)));
                    entreesMelangees[i] = entrees[indices.get(i)];
                }
                images = imagesMelangees;
                entrees = entreesMelangees;
            }

            for (int i = 0; i < images.size(); i++) {
                float[] entree = entrees[i];
                int labelReel = images.get(i).label();

                for (int k = 0; k < nbNeuronesAEntrainer; k++) {
                    coucheNeurones[k].metAJour(entree);

                    // Stratégie one-vs-all : cible 1.0 pour la classe k, 0.0 pour les autres
                    float cible = (labelReel == LABELS[k]) ? 1.0f : 0.0f;
                    float delta = cible - coucheNeurones[k].sortie();
                    sommeErreursCarrees += delta * delta;

                    float[] synapses = coucheNeurones[k].synapses();
                    for (int j = 0; j < entree.length; j++)
                        synapses[j] += entree[j] * ETA * delta;
                    coucheNeurones[k].fixeBiais(coucheNeurones[k].biais() + ETA * delta);
                }
            }

            globaleMse = sommeErreursCarrees / (images.size() * nbNeuronesAEntrainer);
            double dureeIterMs = (System.nanoTime() - tempsDebutIter) / 1_000_000.0;
            System.out.printf("  [Itération : %5d ──── MSE : %.6f ──── Durée : %6.2f ms]\n", iter, globaleMse,
                    dureeIterMs);
            iter++;
        }

        System.out.printf("\n Fin du processus. Itérations exécutées : %d | Durée globale : %.2f s.\n",
                iter, (System.currentTimeMillis() - tempsDebutTotal) / 1000.0);

        sauvegarderConfiguration();
        if (modeMonoNeuroneActif) {
            coucheNeurones[0].sauvegarde(saveDir() + "neurone_binaire_cat.txt");
        } else {
            for (int k = 0; k < CLASSES.length; k++)
                coucheNeurones[k].sauvegarde(saveDir() + "neurone_" + CLASSES[k] + ".txt");
        }
        System.out.println(" Configuration, poids et biais structurels sauvés dans : " + saveDir());
        System.out.println("\n Performances instantanées mesurées sur le Dataset d'Entraînement :");
        afficherMatriceEtCalculs(images, entrees, coucheNeurones);
    }

    // =========================================================
    // TEST
    // =========================================================

    /**
     * Charge les poids entraînés et mesure les performances sur le dataset de test.
     */
    static void test() throws Exception {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" PHASE D'ÉVALUATION (TEST)");
        System.out.println("════════════════════════════════════════════════════════");

        List<Image> images = chargerImages(DATASET_TEST);
        if (images.isEmpty()) {
            System.err.println(" Annulation : aucune donnée de test.");
            return;
        }

        float[][] entrees = chargerEntrees(images);
        int nbEntrees = entrees[0].length;
        int nbNeuronesACharger = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] neurones = new Neurone[nbNeuronesACharger];

        try {
            if (modeMonoNeuroneActif) {
                neurones[0] = creerNeurone(nbEntrees);
                neurones[0].chargement(saveDir() + "neurone_binaire_cat.txt");
            } else {
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k] = creerNeurone(nbEntrees);
                    neurones[k].chargement(saveDir() + "neurone_" + CLASSES[k] + ".txt");
                }
            }
        } catch (Exception e) {
            System.err.println("\n[ERREUR CRITIQUE] Impossible de charger les poids depuis " + saveDir());
            System.err.println("Détail de l'erreur : " + e.getMessage());
            System.err.println(
                    "-> Assurez-vous d'avoir lancé un ENTRAÎNEMENT complet avec la MÊME configuration (Couleur/Traitement) avant d'évaluer.\n");
            return;
        }

        System.out.println("\n Performances structurelles mesurées sur le Dataset de Test :");
        afficherMatriceEtCalculs(images, entrees, neurones);
        System.out.println("\n Évaluation terminée.");
    }

    /**
     * Calcule et affiche la matrice de confusion et la précision par classe.
     * Binaire : seuil à 0.5 (ou > 0 pour Heavyside).
     * Multiclasse : winner-takes-all sur les sorties des 3 neurones.
     */
    private static void afficherMatriceEtCalculs(List<Image> images, float[][] entrees, Neurone[] neurones) {
        int colsMatrice = modeMonoNeuroneActif ? 2 : CLASSES.length;
        int[][] confusion = new int[CLASSES.length][colsMatrice];
        int nbCorrects = 0;

        for (int i = 0; i < images.size(); i++) {
            float[] entree = entrees[i];
            int prediction = 0;

            if (modeMonoNeuroneActif) {
                neurones[0].metAJour(entree);
                float score = neurones[0].sortie();
                prediction = (score >= 0.5f || (typeNeuroneActif.equalsIgnoreCase("heavyside") && score > 0f)) ? 0 : 1;
            } else {
                float maxSortie = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k].metAJour(entree);
                    if (neurones[k].sortie() > maxSortie) {
                        maxSortie = neurones[k].sortie();
                        prediction = k;
                    }
                }
            }

            int labelReel = images.get(i).label();
            int indexReel = 0;
            for (int k = 0; k < LABELS.length; k++) {
                if (LABELS[k] == labelReel) {
                    indexReel = k;
                    break;
                }
            }

            confusion[indexReel][prediction]++;

            if (modeMonoNeuroneActif) {
                if ((indexReel == 0 && prediction == 0) || (indexReel != 0 && prediction == 1))
                    nbCorrects++;
            } else {
                if (prediction == indexReel)
                    nbCorrects++;
            }
        }

        String col = "──────────";
        String sep = "├" + (col + "┼").repeat(colsMatrice) + col + "┤";
        System.out.println("┌" + (col + "┬").repeat(colsMatrice) + col + "┐");
        System.out.printf("│ %-8s │", "Réel\\Pr");
        if (modeMonoNeuroneActif) {
            System.out.printf(" %-8s │ %-8s │\n", "cat", "non-cat");
        } else {
            for (String c : CLASSES)
                System.out.printf(" %-8s │", c);
            System.out.println();
        }
        System.out.println(sep);
        for (int i = 0; i < CLASSES.length; i++) {
            System.out.printf("│ %-8s │", CLASSES[i]);
            for (int j = 0; j < colsMatrice; j++)
                System.out.printf(" %-8d │", confusion[i][j]);
            System.out.println();
        }
        System.out.println("└" + (col + "┴").repeat(colsMatrice) + col + "┘");

        System.out.println("\n Précision analytique par classe cible :");
        for (int k = 0; k < CLASSES.length; k++) {
            int total = 0;
            for (int j = 0; j < colsMatrice; j++)
                total += confusion[k][j];
            int vraisResultats = modeMonoNeuroneActif ? ((k == 0) ? confusion[k][0] : confusion[k][1])
                    : confusion[k][k];
            double prec = total > 0 ? 100.0 * vraisResultats / total : 0.0;
            System.out.printf("    %-6s : %3d / %3d réussis (%.2f%%)\n", CLASSES[k], vraisResultats, total, prec);
        }
        System.out.printf("\n Précision globale du modèle : %d / %d (%.2f%%)\n", nbCorrects, images.size(),
                100.0 * nbCorrects / images.size());
    }

    // =========================================================
    // SORT
    // =========================================================

    /**
     * Classe les images de to_sort/ avec le modèle entraîné et les copie dans les
     * dossiers de destination.
     */
    static void sort() throws Exception {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" DISTRIB-TRI DE FLUX (SORT)");
        System.out.println("════════════════════════════════════════════════════════");

        List<String> fichiers = Image.listeFichiers(DATASET_SORT);
        if (fichiers == null || fichiers.isEmpty()) {
            System.err.println(" Annulation : aucune donnée brute à trier dans " + DATASET_SORT);
            return;
        }

        List<String> imagesATrier = new ArrayList<>();
        for (String f : fichiers) {
            String low = f.toLowerCase();
            if (low.endsWith(".jpg") || low.endsWith(".png"))
                imagesATrier.add(f);
        }
        if (imagesATrier.isEmpty()) {
            System.err.println(" Annulation : Aucun format pris en compte (.jpg, .png) trouvé.");
            return;
        }

        // Nettoyage des anciennes sorties pour ne pas mélanger avec un tri précédent
        String outDir = sortOutDir();
        System.out.println("  Nettoyage des anciens dossiers de tri dans : " + outDir);
        List<String> dossiersANettoyer = new ArrayList<>(Arrays.asList(CLASSES));
        if (modeMonoNeuroneActif)
            dossiersANettoyer.add("non_cat");

        for (String cl : dossiersANettoyer) {
            File folder = new File(outDir + cl);
            if (folder.exists() && folder.isDirectory()) {
                File[] anciensFichiers = folder.listFiles();
                if (anciensFichiers != null)
                    for (File f : anciensFichiers)
                        if (f.isFile())
                            f.delete();
            }
        }

        System.out.printf("  %d images détectées dans 'to_sort/'. Début du tri vers '%s'...\n", imagesATrier.size(),
                outDir);

        boolean estGris = typeCouleurActif.equals("GRI");
        List<Image> imagesChargees = new ArrayList<>();
        for (String cheminFichier : imagesATrier)
            imagesChargees.add(new Image(cheminFichier, -1, estGris));
        float[][] entreesSort = chargerEntrees(imagesChargees);

        int nbEntrees = entreesSort[0].length;
        int nbNeuronesACharger = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] neurones = new Neurone[nbNeuronesACharger];

        try {
            if (modeMonoNeuroneActif) {
                neurones[0] = creerNeurone(nbEntrees);
                neurones[0].chargement(saveDir() + "neurone_binaire_cat.txt");
            } else {
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k] = creerNeurone(nbEntrees);
                    neurones[k].chargement(saveDir() + "neurone_" + CLASSES[k] + ".txt");
                }
            }
        } catch (Exception e) {
            System.err.println(" Erreur critique : Fichier(s) de poids manquant(s) dans " + saveDir()
                    + ". Ré-entraînez le réseau.");
            return;
        }

        // dispatch[k] = nombre d'images copiées dans le dossier de la classe k
        // Index CLASSES.length réservé à "non_cat" en mode binaire
        int[] dispatch = new int[CLASSES.length + (modeMonoNeuroneActif ? 1 : 0)];

        for (int i = 0; i < imagesChargees.size(); i++) {
            Image img = imagesChargees.get(i);
            float[] entree = entreesSort[i];
            Path source = Paths.get(img.chemin());
            String nomFichier = source.getFileName().toString();
            Path cible;

            if (modeMonoNeuroneActif) {
                neurones[0].metAJour(entree);
                float score = neurones[0].sortie();
                boolean estCat = (score >= 0.5f || (typeNeuroneActif.equalsIgnoreCase("heavyside") && score > 0f));
                if (estCat) {
                    cible = Paths.get(outDir + "cat/" + nomFichier);
                    dispatch[0]++;
                } else {
                    cible = Paths.get(outDir + "non_cat/" + nomFichier);
                    dispatch[CLASSES.length]++;
                }
            } else {
                int gagnant = 0;
                float maxSortie = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k].metAJour(entree);
                    if (neurones[k].sortie() > maxSortie) {
                        maxSortie = neurones[k].sortie();
                        gagnant = k;
                    }
                }
                cible = Paths.get(outDir + CLASSES[gagnant] + "/" + nomFichier);
                dispatch[gagnant]++;
            }

            if (!Files.exists(cible.getParent()))
                Files.createDirectories(cible.getParent());
            Files.copy(source, cible, StandardCopyOption.REPLACE_EXISTING);
        }

        System.out.println("\n Tri fini avec succès ! Synthèse du flux filtré :");
        if (modeMonoNeuroneActif) {
            System.out.printf("  └─> [cat]     : %d images copiées dans '%s'.\n", dispatch[0], outDir + "cat/");
            System.out.printf("  └─> [non_cat] : %d images copiées dans '%s'.\n", dispatch[CLASSES.length],
                    outDir + "non_cat/");
        } else {
            for (int k = 0; k < CLASSES.length; k++)
                System.out.printf("  └─> [%-5s]   : %d images copiées dans '%s'.\n", CLASSES[k], dispatch[k],
                        outDir + CLASSES[k] + "/");
        }
    }

    // =========================================================
    // CLEAR
    // =========================================================

    static void clear(Scanner sc) {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" NETTOYAGE DES SORTIES DE TRI (CLEAR)");
        System.out.println("════════════════════════════════════════════════════════");

        String[] optionsClear = {
                "Binaire uniquement (" + DATASET_SORT_OUT_BINAIRE + ")",
                "Multiple uniquement (" + DATASET_SORT_OUT_MULTIPLE + ")",
                "Les deux (Binaire + Multiple)"
        };
        String choixClear = menuChoix(sc, "Que voulez-vous nettoyer ?", optionsClear);

        List<String> dirsANettoyer = new ArrayList<>();
        if (choixClear.startsWith("Binaire"))
            dirsANettoyer.add(DATASET_SORT_OUT_BINAIRE);
        if (choixClear.startsWith("Multiple"))
            dirsANettoyer.add(DATASET_SORT_OUT_MULTIPLE);
        if (choixClear.startsWith("Les deux")) {
            dirsANettoyer.add(DATASET_SORT_OUT_BINAIRE);
            dirsANettoyer.add(DATASET_SORT_OUT_MULTIPLE);
        }

        List<String> sousDossiers = new ArrayList<>(Arrays.asList(CLASSES));
        sousDossiers.add("non_cat");

        int totalSupprime = 0;
        for (String dir : dirsANettoyer) {
            System.out.println("\n  Nettoyage de : " + dir);
            for (String sd : sousDossiers) {
                File dossier = new File(dir + sd);
                if (!dossier.exists() || !dossier.isDirectory())
                    continue;
                File[] fichiers = dossier.listFiles();
                int count = 0;
                if (fichiers != null)
                    for (File f : fichiers)
                        if (f.isFile()) {
                            f.delete();
                            count++;
                        }
                totalSupprime += count;
                System.out.printf("    └─> [%-8s] : %d fichier(s) supprimé(s)\n", sd, count);
            }
        }
        System.out.printf("\n Total supprimé : %d fichier(s). Nettoyage terminé.\n", totalSupprime);
    }
}
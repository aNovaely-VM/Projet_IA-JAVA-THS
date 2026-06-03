import java.io.*;
import java.nio.file.*;
import java.util.*;

/*
 * Regarder README.md pour les instructions
 */
public class Compilation 
{
    // =========================================================
    //  VARIABLES DE CONFIGURATION
    // =========================================================

    // Chemins vers les datasets
    static final String DATASET_TRAIN = "dataset_groupe_6/train/";
    static final String DATASET_TEST  = "dataset_groupe_6/test/";
    static final String DATASET_SORT  = "dataset_groupe_6/naming/to_sort/"; 

    // Classes à reconnaître et leurs labels associés
    static final String[] CLASSES = {"cat", "dog", "wild"};
    static final int[]    LABELS  = {0,      1,     2};

    // Taille de resize (64 => image 64x64)
    static final int RESIZE = 64;

    // Options disponibles
    static final String[] OPTIONS_COULEUR = {"GRI", "COL", "TSL"};
    static final String[] OPTIONS_NEURONE = {"heavyside", "relu", "sigmoide"};
    static final String[] OPTIONS_TRAITEMENT = {"simple", "mirror", "histogramme (Oblige COL ou TSL)", "HOG (Oblige GRI)"};
    static final String[] OPTIONS_ARCHITECTURE = {"1 Neurone (Cat ou non - Binaire)", "3 Neurones (Cat, Dog, Wild - Multiclasse)"};
    static final String[] OPTIONS_ORDRE = {"Ordre séquentiel (Linéaire)", "Mélange aléatoire (Shuffle)"};

    // Choix actifs (sélectionnés par l'utilisateur au runtime)
    static String typeCouleurActif = "GRI";       
    static String typeNeuroneActif = "sigmoide";  
    static String typeTraitementActif = "simple"; 
    static boolean modeMonoNeuroneActif = false;  
    static boolean modeShuffleActif = false; 

    // Seuil de convergence MSE, Taux d'apprentissage et Limite d'itérations
    static final float MSE_LIMITE = 0.01f;
    static final float ETA = 0.0001f;
    static final int MAX_ITERATIONS = 200; 

    // Dossier de sauvegarde des logs et poids
    static final String SAVE_DIR = "Logs/";

    // SYSTEME DE CACHE POUR LES IMAGES TRANSFORMEES
    private static final Map<Integer, float[]> cacheTransformations = new HashMap<>();

    // =========================================================
    //  MAIN
    // =========================================================
    public static void main(String[] args) throws Exception 
    {
        System.out.println("┌──────────────────────────────────────────┐");
        System.out.println("│  Compilation.java ─ Réseau de neurones   │");
        System.out.println("└──────────────────────────────────────────┘\n");

        Scanner sc = new Scanner(System.in);

        // --- 1. DEMANDE D'ABORD L'ACTION À EFFECTUER ---
        System.out.println("Que voulez-vous faire ?");
        System.out.println("  1) Entraîner puis évaluer (Train + Test)");
        System.out.println("  2) Entraîner puis trier (Train + Sort)");
        System.out.println("  3) Entraîner uniquement (Train only)");
        System.out.println("  4) Évaluer uniquement (Test only ─ ! Nécessite un modèle déjà entraîné ! )");
        System.out.println("  5) Trier uniquement (Sort only ─ ! Nécessite un modèle déjà entraîné ! )");
        System.out.print("\nVotre choix [1-5] : ");
        String choix = sc.nextLine().trim();
        System.out.println();

        boolean necessiteEntrainement = choix.equals("1") || choix.equals("2") || choix.equals("3");

        // --- 2. CONFIGURATION DYNAMIQUE ---
        if (necessiteEntrainement) {
            typeNeuroneActif = menuChoix(sc, "Choisissez le type de neurone :", OPTIONS_NEURONE);
            
            String choixTraitementRaw = menuChoix(sc, "Choisissez le type de traitement :", OPTIONS_TRAITEMENT);
            if (choixTraitementRaw.contains(" ")) {
                typeTraitementActif = choixTraitementRaw.split(" ")[0].trim();
            } else {
                typeTraitementActif = choixTraitementRaw.trim();
            }
            
            if (typeTraitementActif.equalsIgnoreCase("HOG")) {
                System.out.println("[INFO] Le mode HOG requiert des niveaux de gris. Configuration automatique sur 'GRI'.\n");
                typeCouleurActif = "GRI";
            } else if (typeTraitementActif.equalsIgnoreCase("histogramme")) {
                System.out.println("[INFO] Le mode Histogramme requiert de la couleur. Configuration automatique sur 'COL'.\n");
                typeCouleurActif = "COL";
            } else {
                typeCouleurActif = menuChoix(sc, "Choisissez le mode couleur :", OPTIONS_COULEUR);
            }

            String choixArch = menuChoix(sc, "Choisissez l'architecture du réseau :", OPTIONS_ARCHITECTURE);
            modeMonoNeuroneActif = choixArch.startsWith("1");

            String choixOrdre = menuChoix(sc, "Choisissez l'ordre des images :", OPTIONS_ORDRE);
            modeShuffleActif = choixOrdre.contains("Shuffle");

            cacheTransformations.clear();
            afficherConfig();
        } else {
            // Mode évaluation/tri direct : Chargement automatique des logs de configuration passés
            try {
                chargerConfiguration();
                System.out.println("[INFO] Logs de configuration passée chargés avec succès.");
            } catch (IOException e) {
                System.err.println("[ATTENTION] Aucun fichier de configuration globale trouvé dans " + SAVE_DIR + "config.txt");
                System.err.println("[ATTENTION] Tentative de détection adaptative sommaire...");
                
                File modCat = new File(SAVE_DIR + "neurone_cat.txt");
                File modDog = new File(SAVE_DIR + "neurone_dog.txt");
                File modWild = new File(SAVE_DIR + "neurone_wild.txt");
                
                if (modCat.exists() && modDog.exists() && modWild.exists()) {
                    modeMonoNeuroneActif = false;
                    System.out.println("[INFO] Modèles multiclasses détectés (cat, dog, wild).");
                } else {
                    modeMonoNeuroneActif = true;
                    System.out.println("[INFO] Modèles multiclasses manquants. Utilisation par défaut du neurone binaire.");
                }
            }
            afficherConfig();
        }

        // --- 3. EXECUTION DE L'ACTION ---
        switch (choix) {
            case "1" -> { train(); test(); }
            case "2" -> { train(); sort(); }
            case "3" -> train();
            case "4" -> test();
            case "5" -> sort();
            default  -> System.out.println("Choix invalide. Fin du programme.");
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
            } catch (NumberFormatException e) { /* continue */ }
            System.out.println("Choix invalide. Réessayez.\n");
        }
    }

    static void afficherConfig() 
    {
        System.out.println("┌────────────────────────────────────────────────────────┐");
        System.out.println("│                 CONFIGURATION ACTIVE                   │");
        System.out.println("├────────────────────────────────────────────────────────┤");
        System.out.printf("│  Dataset train    : %-34s │\n", DATASET_TRAIN);
        System.out.printf("│  Dataset test     : %-34s │\n", DATASET_TEST);
        System.out.printf("│  Dataset à trier  : %-34s │\n", DATASET_SORT);
        System.out.printf("│  Architecture     : %-34s │\n", (modeMonoNeuroneActif ? "1 Neurone (Cat / Non-Cat)" : "3 Neurones (Collaboratif)"));
        System.out.printf("│  Neurone choisi   : %-34s │\n", typeNeuroneActif);
        System.out.printf("│  Mode image       : %-34s │\n", typeCouleurActif);
        System.out.printf("│  Traitement       : %-34s │\n", typeTraitementActif);
        System.out.printf("│  Dimension d'adm. : %-34s │\n", (RESIZE + "x" + RESIZE));
        System.out.printf("│  MSE cible        : %-34s │\n", MSE_LIMITE);
        System.out.printf("│  Max Itérations   : %-34s │\n", MAX_ITERATIONS);
        System.out.printf("│  Pas d'app. (Eta) : %-34s │\n", ETA);
        System.out.println("└────────────────────────────────────────────────────────┘\n");
    }

    // =========================================================
    //  SAUVEGARDE ET CHARGEMENT DE LA CONFIGURATION
    // =========================================================
    private static void sauvegarderConfiguration() throws IOException {
        File dir = new File(SAVE_DIR);
        if (!dir.exists()) dir.mkdirs();
        
        try (PrintWriter pw = new PrintWriter(new FileWriter(SAVE_DIR + "config.txt"))) {
            pw.println("typeCouleurActif=" + typeCouleurActif);
            pw.println("typeNeuroneActif=" + typeNeuroneActif);
            pw.println("typeTraitementActif=" + typeTraitementActif);
            pw.println("modeMonoNeuroneActif=" + modeMonoNeuroneActif);
            pw.println("modeShuffleActif=" + modeShuffleActif);
        }
    }

    private static void chargerConfiguration() throws IOException {
        File configFile = new File(SAVE_DIR + "config.txt");
        if (!configFile.exists()) {
            throw new FileNotFoundException("Fichier de configuration absent.");
        }
        
        try (BufferedReader br = new BufferedReader(new FileReader(configFile))) {
            String ligne;
            while ((ligne = br.readLine()) != null) {
                String[] parts = ligne.split("=");
                if (parts.length == 2) {
                    String cle = parts[0].trim();
                    String valeur = parts[1].trim();
                    switch (cle) {
                        case "typeCouleurActif" -> typeCouleurActif = valeur;
                        case "typeNeuroneActif" -> typeNeuroneActif = valeur;
                        case "typeTraitementActif" -> typeTraitementActif = valeur;
                        case "modeMonoNeuroneActif" -> modeMonoNeuroneActif = Boolean.parseBoolean(valeur);
                        case "modeShuffleActif" -> modeShuffleActif = Boolean.parseBoolean(valeur);
                    }
                }
            }
        }
    }

    // =========================================================
    //  PIPELINE DE TRAITEMENT D'IMAGES
    // =========================================================
    static float[][] chargerEntrees(List<Image> images) 
    {
        if (images.isEmpty()) return new float[0][0];
        int tailleVecteur = obtenirVecteurTransforme(images.get(0)).length;
        float[][] entrees = new float[images.size()][tailleVecteur];
        
        for (int i = 0; i < images.size(); i++) {
            entrees[i] = obtenirVecteurTransforme(images.get(i));
        }
        return entrees;
    }

    static float[] obtenirVecteurTransforme(Image img) {
        Integer cleUnique = img.hashCode(); 
        
        if (cacheTransformations.containsKey(cleUnique)) {
            return cacheTransformations.get(cleUnique);
        }
        float[] transforme = appliquerTransformations(img);
        cacheTransformations.put(cleUnique, transforme);
        return transforme;
    }

    static float[] appliquerTransformations(Image img) {
        int[] pixelsBruts = img.donnees();
        float[] donneesModifiees;
        int largeur = img.largeur();
        int hauteur = img.hauteur();

        switch (typeCouleurActif) {
            case "GRI" -> {
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
                donneesModifiees = new float[pixelsBruts.length * (img.estEnNiveauxDeGris() ? 1 : 3)];
                if (img.estEnNiveauxDeGris()) {
                    for (int i = 0; i < pixelsBruts.length; i++) {
                        donneesModifiees[i] = pixelsBruts[i] / 255.f;
                    }
                } else {
                    int index = 0;
                    for (int i = 0; i < pixelsBruts.length; i++) {
                        int p = pixelsBruts[i];
                        donneesModifiees[index++] = ((p >> 16) & 0xFF) / 255.f;
                        donneesModifiees[index++] = ((p >> 8) & 0xFF) / 255.f;
                        donneesModifiees[index++] = (p & 0xFF) / 255.f;
                    }
                }
            }
        }

        boolean estMulticomposante = typeCouleurActif.equals("COL") || typeCouleurActif.equals("TSL");
        
        switch (typeTraitementActif.toLowerCase()) {
            case "mirror" -> donneesModifiees = appliquerEffetMiroir(donneesModifiees, largeur, hauteur, estMulticomposante);
            case "histogramme" -> donneesModifiees = appliquerEgalisationHistogramme(donneesModifiees, estMulticomposante);
            case "hog" -> donneesModifiees = calculerHOG(donneesModifiees, largeur, hauteur, estMulticomposante);
        }

        return donneesModifiees;
    }

    private static float[] convertRGBtoTSL(int[] rbgColor, boolean estGris) {
        if (estGris) {
            float[] tsl = new float[rbgColor.length * 3];
            for (int i = 0; i < rbgColor.length; i++) {
                float val = rbgColor[i] / 255.f;
                tsl[3 * i] = 0.0f;
                tsl[3 * i + 1] = 0.0f;
                tsl[3 * i + 2] = val;
            }
            return tsl;
        }
        
        float[] tsl = new float[rbgColor.length * 3];
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
                if (max == r) {
                    h = (g - b) / d + (g < b ? 6.0f : 0.0f);
                } else if (max == g) {
                    h = (b - r) / d + 2.0f;
                } else if (max == b) {
                    h = (r - g) / d + 4.0f;
                }
                h /= 6.0f;
            }
            tsl[3 * i] = h;
            tsl[3 * i + 1] = s;
            tsl[3 * i + 2] = l;
        }
        return tsl;
    }

    private static float[] appliquerEffetMiroir(float[] src, int large, int haut, boolean multicomposante) {
        float[] dest = new float[src.length];
        int pas = multicomposante ? 3 : 1;
        for (int y = 0; y < haut; y++) {
            for (int x = 0; x < large; x++) {
                int idxSrc = (y * large + x) * pas;
                int idxDest = (y * large + (large - 1 - x)) * pas;
                for (int c = 0; c < pas; c++) {
                    dest[idxDest + c] = src[idxSrc + c];
                }
            }
        }
        return dest;
    }

    private static float[] appliquerEgalisationHistogramme(float[] src, boolean multicomposante) {
        float[] dest = new float[src.length];
        int bins = 256;
        int[] histogramme = new int[bins];

        if (!multicomposante) {
            for (float val : src) {
                int bin = Math.min(bins - 1, Math.max(0, (int) (val * 255)));
                histogramme[bin]++;
            }
            int[] cdf = new int[bins];
            cdf[0] = histogramme[0];
            for (int i = 1; i < bins; i++) cdf[i] = cdf[i - 1] + histogramme[i];

            int cdfMin = 0;
            for (int i = 0; i < bins; i++) {
                if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
            }

            for (int i = 0; i < src.length; i++) {
                int bin = Math.min(bins - 1, Math.max(0, (int) (src[i] * 255)));
                dest[i] = ((float) (cdf[bin] - cdfMin) / (src.length - cdfMin));
            }
        } else {
            System.arraycopy(src, 0, dest, 0, src.length);
            int totalPixels = src.length / 3;
            boolean estTSL = typeCouleurActif.equals("TSL");

            for (int i = 0; i < totalPixels; i++) {
                float v = estTSL ? src[3 * i + 2] : (src[3 * i] + src[3 * i + 1] + src[3 * i + 2]) / 3.0f; 
                int bin = Math.min(bins - 1, Math.max(0, (int) (v * 255)));
                histogramme[bin]++;
            }
            int[] cdf = new int[bins];
            cdf[0] = histogramme[0];
            for (int i = 1; i < bins; i++) cdf[i] = cdf[i - 1] + histogramme[i];
            
            int cdfMin = 0;
            for (int i = 0; i < bins; i++) {
                if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
            }

            for (int i = 0; i < totalPixels; i++) {
                float vOriginale = estTSL ? src[3 * i + 2] : (src[3 * i] + src[3 * i + 1] + src[3 * i + 2]) / 3.0f;
                int bin = Math.min(bins - 1, Math.max(0, (int) (vOriginale * 255)));
                float vEgalisee = ((float) (cdf[bin] - cdfMin) / (totalPixels - cdfMin));
                
                if (estTSL) {
                    dest[3 * i + 2] = vEgalisee;
                } else { 
                    float facteur = vEgalisee / (vOriginale + 0.001f);
                    dest[3 * i] = Math.min(1.0f, src[3 * i] * facteur);
                    dest[3 * i + 1] = Math.min(1.0f, src[3 * i + 1] * facteur);
                    dest[3 * i + 2] = Math.min(1.0f, src[3 * i + 2] * facteur);
                }
            }
        }
        return dest;
    }

    private static float[] calculerHOG(float[] src, int large, int haut, boolean multicomposante) {
        float[][] intensite = new float[haut][large];
        int pas = multicomposante ? 3 : 1;
        for (int y = 0; y < haut; y++) {
            for (int x = 0; x < large; x++) {
                int idx = (y * large + x) * pas;
                if (multicomposante) {
                    intensite[y][x] = 0.299f * src[idx] + 0.587f * src[idx + 1] + 0.114f * src[idx + 2];
                } else {
                    intensite[y][x] = src[idx];
                }
            }
        }

        float[][] magnitude = new float[haut][large];
        float[][] orientation = new float[haut][large];

        for (int y = 1; y < haut - 1; y++) {
            for (int x = 1; x < large - 1; x++) {
                float gx = intensite[y][x + 1] - intensite[y][x - 1];
                float gy = intensite[y + 1][x] - intensite[y - 1][x];
                magnitude[y][x] = (float) Math.sqrt(gx * gx + gy * gy);
                float angle = (float) Math.toDegrees(Math.atan2(gy, gx));
                if (angle < 0) angle += 180.0f;
                orientation[y][x] = angle;
            }
        }

        int tailleCellule = 8;
        int numBins = 9;
        int cellulesX = large / tailleCellule;
        int cellulesY = haut / tailleCellule;
        float[] vecteurHOG = new float[cellulesX * cellulesY * numBins];

        int indexVecteur = 0;
        for (int cy = 0; cy < cellulesY; cy++) {
            for (int cx = 0; cx < cellulesX; cx++) {
                float[] histCellule = new float[numBins];
                for (int y = 0; y < tailleCellule; y++) {
                    for (int x = 0; x < tailleCellule; x++) {
                        int px = cx * tailleCellule + x;
                        int py = cy * tailleCellule + y;
                        float mag = magnitude[py][px];
                        float ang = orientation[py][px];

                        int bin = (int) (ang / (180.0f / numBins));
                        if (bin >= numBins) bin = numBins - 1;
                        histCellule[bin] += mag;
                    }
                }
                for (float v : histCellule) {
                    vecteurHOG[indexVecteur++] = v;
                }
            }
        }
        return vecteurHOG; 
    }

    static List<Image> chargerImages(String baseDir) 
    {
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
                if (!low.endsWith(".jpg") && !low.endsWith(".png")) continue;
                liste.add(new Image(chemin, LABELS[c], estGris));
                count++;
            }
            System.out.printf("  Vectorisation | %-5s : %d fichiers chargés\n", CLASSES[c], count);
        }
        return liste;
    }

    static Neurone creerNeurone(int nbEntrees) 
    {
        return switch (typeNeuroneActif.toLowerCase()) {
            case "heavyside" -> new NeuroneHeavyside(nbEntrees);
            case "relu"      -> new NeuroneReLu(nbEntrees);
            default          -> new NeuroneSigmoide(nbEntrees);
        };
    }

    // =========================================================
    //  ENTRAÎNEMENT CONJOINT
    // =========================================================
    static void train() throws Exception 
    {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" PHASE D'ENTRAÎNEMENT");
        System.out.println("════════════════════════════════════════════════════════");
        List<Image> images = chargerImages(DATASET_TRAIN);
        if (images.isEmpty()) { System.err.println(" Annulation : aucune donnée d'entraînement."); return; }

        if (modeShuffleActif) {
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
        for (int k = 0; k < nbNeuronesAEntrainer; k++) {
            coucheNeurones[k] = creerNeurone(nbEntrees);
        }

        System.out.printf(" Lancement de l'optimisation (%d neurone(s) en compétition)...\n", nbNeuronesAEntrainer);
        
        long tempsDebutTotal = System.currentTimeMillis(); 
        double globaleMse = 1.0;
        int iter = 0;

        while (globaleMse > MSE_LIMITE && iter < MAX_ITERATIONS) {
            long tempsDebutIter = System.nanoTime(); 
            double sommeErreursCarrees = 0.0;

            for (int i = 0; i < images.size(); i++) {
                float[] entree = entrees[i]; 
                int labelReel = images.get(i).label();

                for (int k = 0; k < nbNeuronesAEntrainer; k++) {
                    coucheNeurones[k].metAJour(entree);
                    
                    float cible = (labelReel == LABELS[k]) ? 1.0f : 0.0f;
                    float delta = cible - coucheNeurones[k].sortie();
                    
                    sommeErreursCarrees += (delta * delta);

                    float[] synapses = coucheNeurones[k].synapses();
                    for (int j = 0; j < entree.length; j++) {
                        synapses[j] += entree[j] * ETA * delta;
                    }
                    coucheNeurones[k].fixeBiais(coucheNeurones[k].biais() + ETA * delta);
                }
            }

            globaleMse = sommeErreursCarrees / (images.size() * nbNeuronesAEntrainer);
            long tempsFinIter = System.nanoTime(); 
            double dureeIterMs = (tempsFinIter - tempsDebutIter) / 1_000_000.0;

            System.out.printf("  [Itération : %5d ──── MSE : %.6f ──── Durée : %6.2f ms]\n", iter, globaleMse, dureeIterMs);
            iter++;
        }

        long tempsFinTotal = System.currentTimeMillis(); 
        double dureeTotaleSec = (tempsFinTotal - tempsDebutTotal) / 1000.0;
        System.out.printf("\n Fin du processus. Itérations exécutées : %d | Durée globale : %.2f s.\n", iter, dureeTotaleSec);

        // Sauvegarde de la configuration globale
        sauvegarderConfiguration();

        if (modeMonoNeuroneActif) {
            coucheNeurones[0].sauvegarde(SAVE_DIR + "neurone_binaire_cat.txt");
        } else {
            for (int k = 0; k < CLASSES.length; k++) {
                coucheNeurones[k].sauvegarde(SAVE_DIR + "neurone_" + CLASSES[k] + ".txt");
            }
        }
        System.out.println(" Configuration, poids et biais structurels sauvés dans : " + SAVE_DIR);
        
        System.out.println("\n Performances instantanées mesurées sur le Dataset d'Entraînement :");
        afficherMatriceEtCalculs(images, entrees, coucheNeurones);
    }

    // =========================================================
    //  TEST
    // =========================================================
    static void test() throws Exception 
    {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" PHASE D'ÉVALUATION (TEST)");
        System.out.println("════════════════════════════════════════════════════════");
        List<Image> images = chargerImages(DATASET_TEST);
        if (images.isEmpty()) { System.err.println(" Annulation : aucune donnée de test."); return; }

        float[][] entrees = chargerEntrees(images);
        int nbEntrees = entrees[0].length;

        int nbNeuronesACharger = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] neurones = new Neurone[nbNeuronesACharger];
        
        try {
            if (modeMonoNeuroneActif) {
                neurones[0] = creerNeurone(nbEntrees);
                neurones[0].chargement(SAVE_DIR + "neurone_binaire_cat.txt");
            } else {
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k] = creerNeurone(nbEntrees);
                    neurones[k].chargement(SAVE_DIR + "neurone_" + CLASSES[k] + ".txt");
                }
            }
        } catch (Exception e) {
            System.err.println("\n[ERREUR CRITIQUE] Impossible de charger les poids.");
            System.err.println("Détail de l'erreur : " + e.getMessage());
            System.err.println("-> Assurez-vous d'avoir lancé un ENTRAÎNEMENT complet avec la MÊME configuration (Couleur/Traitement) avant d'évaluer.\n");
            return; 
        }

        System.out.println("\n Performances structurelles mesurées sur le Dataset de Test :");
        afficherMatriceEtCalculs(images, entrees, neurones);
        System.out.println("\n Évaluation terminée.");
    }
    
    private static void afficherMatriceEtCalculs(List<Image> images, float[][] entrees, Neurone[] neurones) {
        int colsMatrice = modeMonoNeuroneActif ? 2 : CLASSES.length; 
        int[][] confusion = new int[CLASSES.length][colsMatrice];
        int nbCorrects = 0;

        for (int i = 0; i < images.size(); i++) {
            float[] entree = entrees[i];
            Image img = images.get(i);
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

            int labelReel = img.label();
            int indexReel = 0;
            for (int k = 0; k < LABELS.length; k++) {
                if (LABELS[k] == labelReel) { indexReel = k; break; }
            }

            confusion[indexReel][prediction]++;
            
            if (modeMonoNeuroneActif) {
                if ((indexReel == 0 && prediction == 0) || (indexReel != 0 && prediction == 1)) nbCorrects++;
            } else {
                if (prediction == indexReel) nbCorrects++;
            }
        }

        String séparateurLigne = "├──────────┼" + "──────────┼".repeat(colsMatrice);
        String bordureHaute   = "┌──────────┬" + "──────────┬".repeat(colsMatrice);
        String bordureBasse   = "└──────────┴" + "──────────┴".repeat(colsMatrice);
        
        séparateurLigne = séparateurLigne.substring(0, séparateurLigne.length() - 1) + "┤";
        bordureHaute   = bordureHaute.substring(0, bordureHaute.length() - 1) + "┐";
        bordureBasse   = bordureBasse.substring(0, bordureBasse.length() - 1) + "┘";

        System.out.println(bordureHaute);
        System.out.printf("│ %-8s │", "Réel\\Pr");
        if (modeMonoNeuroneActif) {
            System.out.printf(" %-8s │ %-8s │\n", "cat", "non-cat");
        } else {
            for (String c : CLASSES) System.out.printf(" %-8s │", c);
            System.out.println();
        }
        System.out.println(séparateurLigne);
        
        for (int i = 0; i < CLASSES.length; i++) {
            System.out.printf("│ %-8s │", CLASSES[i]);
            for (int j = 0; j < colsMatrice; j++) {
                System.out.printf(" %-8d │", confusion[i][j]);
            }
            System.out.println();
        }
        System.out.println(bordureBasse);

        System.out.println("\n Précision analytique par classe cible :");
        for (int k = 0; k < CLASSES.length; k++) {
            int total = 0;
            for (int j = 0; j < colsMatrice; j++) total += confusion[k][j];
            
            double prec;
            if (modeMonoNeuroneActif) {
                int vraisResultats = (k == 0) ? confusion[k][0] : confusion[k][1];
                prec = total > 0 ? 100.0 * vraisResultats / total : 0.0;
                System.out.printf("    %-6s : %3d / %3d réussis (%.2f%%)\n", CLASSES[k], vraisResultats, total, prec);
            } else {
                prec = total > 0 ? 100.0 * confusion[k][k] / total : 0.0;
                System.out.printf("    %-6s : %3d / %3d réussis (%.2f%%)\n", CLASSES[k], confusion[k][k], total, prec);
            }
        }
        System.out.printf("\n Précision globale du modèle : %d / %d (%.2f%%)\n",
            nbCorrects, images.size(), 100.0 * nbCorrects / images.size());
    }

    // =========================================================
    //  SORT (NETTOYAGE ET COPIE INTÉGRÉS)
    // =========================================================
    static void sort() throws Exception 
    {
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
            if (low.endsWith(".jpg") || low.endsWith(".png")) imagesATrier.add(f);
        }

        if (imagesATrier.isEmpty()) {
            System.err.println(" Annulation : Aucun format pris en compte (.jpg, .png) trouvé.");
            return;
        }

        System.out.println("  Nettoyage des anciens dossiers de tri...");
        List<String> dossiersANettoyer = new ArrayList<>(Arrays.asList(CLASSES));
        if (modeMonoNeuroneActif) dossiersANettoyer.add("non_cat");

        for (String cl : dossiersANettoyer) {
            File folder = new File(DATASET_SORT + cl);
            if (folder.exists() && folder.isDirectory()) {
                File[] anciensFichiers = folder.listFiles();
                if (anciensFichiers != null) {
                    for (File f : anciensFichiers) {
                        if (f.isFile()) f.delete();
                    }
                }
            }
        }

        System.out.printf("  %d images détectées dans 'to_sort/'. Début du tri...\n", imagesATrier.size());

        boolean estGris = typeCouleurActif.equals("GRI");
        Image imgTemoin = new Image(imagesATrier.get(0), -1, estGris);
        int nbEntrees = obtenirVecteurTransforme(imgTemoin).length;

        int nbNeuronesACharger = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] neurones = new Neurone[nbNeuronesACharger];
        
        try {
            if (modeMonoNeuroneActif) {
                neurones[0] = creerNeurone(nbEntrees);
                neurones[0].chargement(SAVE_DIR + "neurone_binaire_cat.txt");
            } else {
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k] = creerNeurone(nbEntrees);
                    neurones[k].chargement(SAVE_DIR + "neurone_" + CLASSES[k] + ".txt");
                }
            }
        } catch (Exception e) {
            System.err.println(" Erreur critique : Fichier(s) de poids manquant(s). Ré-entraînez le réseau.");
            return;
        }

        int[] dispatch = new int[CLASSES.length + (modeMonoNeuroneActif ? 1 : 0)];

        for (String cheminFichier : imagesATrier) {
            Image img = new Image(cheminFichier, -1, estGris);
            float[] entree = obtenirVecteurTransforme(img);
            
            Path source = Paths.get(cheminFichier);
            String nomFichier = source.getFileName().toString();
            Path cible;

            if (modeMonoNeuroneActif) {
                neurones[0].metAJour(entree);
                float score = neurones[0].sortie();
                boolean estCat = (score >= 0.5f || (typeNeuroneActif.equalsIgnoreCase("heavyside") && score > 0f));
                
                if (estCat) {
                    cible = Paths.get(DATASET_SORT + "cat/" + nomFichier);
                    dispatch[0]++;
                } else {
                    cible = Paths.get(DATASET_SORT + "non_cat/" + nomFichier);
                    dispatch[3]++; 
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
                cible = Paths.get(DATASET_SORT + CLASSES[gagnant] + "/" + nomFichier);
                dispatch[gagnant]++;
            }

            if (!Files.exists(cible.getParent())) {
                Files.createDirectories(cible.getParent());
            }
            Files.copy(source, cible, StandardCopyOption.REPLACE_EXISTING);
        }

        System.out.println("\n Tri fini avec succès ! Synthèse du flux filtré :");
        if (modeMonoNeuroneActif) {
            System.out.printf("  └─> [cat]     : %d images copiées.\n", dispatch[0]);
            System.out.printf("  └─> [non_cat] : %d images copiées.\n", dispatch[3]);
        } else {
            for (int k = 0; k < CLASSES.length; k++) {
                System.out.printf("  └─> [%-5s]   : %d images copiées.\n", CLASSES[k], dispatch[k]);
            }
        }
    }
}
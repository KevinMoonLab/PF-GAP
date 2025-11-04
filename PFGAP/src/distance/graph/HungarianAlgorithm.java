package distance.graph;

import java.util.Arrays;

/**
 * Hungarian Algorithm 
 * Input: costMatrix (2D double array), where costMatrix[i][j] is the cost of assigning row i to column j.
 * Output: int[] assignment where assignment[i] = j means row i assigned to column j, or -1 if none.
 *
 * This implementation minimizes total cost.
 */
public class HungarianAlgorithm {

    /**
     * Solves the assignment problem using the Hungarian algorithm.
     * @param costMatrix - 2D array of costs
     * @param sumType - "min" for minimizing total cost, "max" for maximizing (will negate internally)
     * @return assignment array, where index is row, value is assigned column or -1
     */
    public static int[] hgAlgorithm(double[][] costMatrix, String sumType) {
        if ("max".equalsIgnoreCase(sumType)) {
            // Convert to minimization problem by negating costs
            double maxCost = Double.NEGATIVE_INFINITY;
            for (double[] row : costMatrix) {
                for (double c : row) {
                    if (c > maxCost) maxCost = c;
                }
            }
            for (int i = 0; i < costMatrix.length; i++) {
                for (int j = 0; j < costMatrix[i].length; j++) {
                    costMatrix[i][j] = maxCost - costMatrix[i][j];
                }
            }
        }
        return computeAssignment(costMatrix);
    }

    private static int[] computeAssignment(double[][] costMatrix) {
        int nRows = costMatrix.length;
        int nCols = costMatrix[0].length;
        int dim = Math.max(nRows, nCols);

        // Pad cost matrix to be square if necessary
        double[][] cost = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            Arrays.fill(cost[i], 0);
            if (i < nRows) {
                System.arraycopy(costMatrix[i], 0, cost[i], 0, nCols);
            }
        }

        // Step 1: Subtract row minima
        for (int i = 0; i < dim; i++) {
            double rowMin = Double.POSITIVE_INFINITY;
            for (int j = 0; j < dim; j++) {
                if (cost[i][j] < rowMin) rowMin = cost[i][j];
            }
            for (int j = 0; j < dim; j++) {
                cost[i][j] -= rowMin;
            }
        }

        // Step 2: Subtract column minima
        for (int j = 0; j < dim; j++) {
            double colMin = Double.POSITIVE_INFINITY;
            for (int i = 0; i < dim; i++) {
                if (cost[i][j] < colMin) colMin = cost[i][j];
            }
            for (int i = 0; i < dim; i++) {
                cost[i][j] -= colMin;
            }
        }

        int[] assignment = new int[nRows];
        Arrays.fill(assignment, -1);

        // Masks and helpers
        int[] rowCover = new int[dim];
        int[] colCover = new int[dim];
        int[][] mask = new int[dim][dim]; // 0=none,1=starred zero,2=primed zero

        // Step 3: Star zeros
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (cost[i][j] == 0 && rowCover[i] == 0 && colCover[j] == 0) {
                    mask[i][j] = 1; // star
                    rowCover[i] = 1;
                    colCover[j] = 1;
                }
            }
        }

        Arrays.fill(rowCover, 0);
        Arrays.fill(colCover, 0);

        int step = 4;
        int[] zeroRC = new int[2]; // location of uncovered zero
        boolean done = false;

        while (!done) {
            switch (step) {
                case 4:
                    step = step4(cost, mask, rowCover, colCover, zeroRC, dim);
                    break;
                case 5:
                    step = step5(mask, rowCover, colCover, zeroRC, dim);
                    break;
                case 6:
                    step = step6(cost, rowCover, colCover, dim);
                    break;
                case 7:
                    // Construct assignment
                    for (int i = 0; i < nRows; i++) {
                        for (int j = 0; j < nCols; j++) {
                            if (mask[i][j] == 1) {
                                assignment[i] = j;
                            }
                        }
                    }
                    done = true;
                    break;
            }
        }
        return assignment;
    }

    private static int step4(double[][] cost, int[][] mask, int[] rowCover, int[] colCover, int[] zeroRC, int dim) {
        while (true) {
            int[] loc = findUncoveredZero(cost, rowCover, colCover, dim);
            if (loc[0] == -1) {
                return 6; // No uncovered zero found
            } else {
                int row = loc[0];
                int col = loc[1];
                mask[row][col] = 2; // prime it

                int starCol = findStarInRow(mask, row, dim);
                if (starCol != -1) {
                    rowCover[row] = 1;
                    colCover[starCol] = 0;
                } else {
                    zeroRC[0] = row;
                    zeroRC[1] = col;
                    return 5;
                }
            }
        }
    }

    private static int step5(int[][] mask, int[] rowCover, int[] colCover, int[] zeroRC, int dim) {
        int count = 0;
        int[][] path = new int[dim * 2][2];
        path[count][0] = zeroRC[0];
        path[count][1] = zeroRC[1];

        boolean done = false;
        while (!done) {
            int row = findStarInCol(mask, path[count][1], dim);
            if (row != -1) {
                count++;
                path[count][0] = row;
                path[count][1] = path[count - 1][1];
            } else {
                done = true;
                break;
            }

            int col = findPrimeInRow(mask, path[count][0], dim);
            count++;
            path[count][0] = path[count - 1][0];
            path[count][1] = col;
        }

        // Augment path
        for (int i = 0; i <= count; i++) {
            if (mask[path[i][0]][path[i][1]] == 1) {
                mask[path[i][0]][path[i][1]] = 0;
            } else if (mask[path[i][0]][path[i][1]] == 2) {
                mask[path[i][0]][path[i][1]] = 1;
            }
        }

        // Clear covers
        Arrays.fill(rowCover, 0);
        Arrays.fill(colCover, 0);

        // Erase primes
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (mask[i][j] == 2) {
                    mask[i][j] = 0;
                }
            }
        }

        return 4;
    }

    private static int step6(double[][] cost, int[] rowCover, int[] colCover, int dim) {
        double minVal = Double.POSITIVE_INFINITY;
        for (int i = 0; i < dim; i++) {
            if (rowCover[i] == 0) {
                for (int j = 0; j < dim; j++) {
                    if (colCover[j] == 0) {
                        if (cost[i][j] < minVal) {
                            minVal = cost[i][j];
                        }
                    }
                }
            }
        }

        for (int i = 0; i < dim; i++) {
            if (rowCover[i] == 1) {
                for (int j = 0; j < dim; j++) {
                    cost[i][j] += minVal;
                }
            }
        }

        for (int j = 0; j < dim; j++) {
            if (colCover[j] == 0) {
                for (int i = 0; i < dim; i++) {
                    cost[i][j] -= minVal;
                }
            }
        }
        return 4;
    }

    private static int[] findUncoveredZero(double[][] cost, int[] rowCover, int[] colCover, int dim) {
        for (int i = 0; i < dim; i++) {
            if (rowCover[i] == 0) {
                for (int j = 0; j < dim; j++) {
                    if (colCover[j] == 0 && cost[i][j] == 0) {
                        return new int[]{i, j};
                    }
                }
            }
        }
        return new int[]{-1, -1};
    }

    private static int findStarInRow(int[][] mask, int row, int dim) {
        for (int j = 0; j < dim; j++) {
            if (mask[row][j] == 1) return j;
        }
        return -1;
    }

    private static int findStarInCol(int[][] mask, int col, int dim) {
        for (int i = 0; i < dim; i++) {
            if (mask[i][col] == 1) return i;
        }
        return -1;
    }

    private static int findPrimeInRow(int[][] mask, int row, int dim) {
        for (int j = 0; j < dim; j++) {
            if (mask[row][j] == 2) return j;
        }
        return -1;
    }
}


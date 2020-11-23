library(tidyverse)
library(gridExtra)

#setwd

####### HIGHER POLYNOMIALS BOXPLOT #######
#read in file:5 features (like in paper) and higher degree polynomials
results <- read.csv(shortest_path_100n_4_20pol) 

results = results %>% filter(polynomial degree > 8  )

#calculate normalized SPO+ as in paper
results <- results %>% mutate(SPOplus_norm_spo = SPOplus_spoloss_test/zstar_avg_test,
                              LS_norm_spo = LS_spoloss_test/zstar_avg_test,
                              RF_norm_spo = RF_spoloss_test/zstar_avg_test,
                              Absolute_norm_spo = Absolute_spoloss_test/zstar_avg_test)



#Rename for boxplot

colnames(results)[colnames(results)=="SPOplus_norm_spo"] <- "SPO+"
colnames(results)[colnames(results)=="LS_norm_spo"] <- "Least Squares"
colnames(results)[colnames(results)=="RF_norm_spo"] <- "Random Forests"
colnames(results)[colnames(results)=="Absolute_norm_spo"] <- "Absolute Loss"


#keep relevant columns
results_relevant = results %>% 
  select(grid_dim, p_features, n_train, polykernel_degree, polykernel_noise_half_width,
         `SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`)

results_relevant_fixed = results_relevant %>%
  gather(`SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`,
         key = "method", value = "spo")

results_relevant_fixed$method = as.factor(results_relevant_fixed$method)
results_relevant_fixed$polykernel_noise_half_width = as.factor(results_relevant_fixed$polykernel_noise_half_width)

# Labelers

half_width_names <- c(
  '0' = "Noise Half-width = 0",
  '0.5' = "Noise Half-width = 0.5" 
)

polynomial_degree <- c(
  '1' = "Polynomial Degree = 1",
  '4' = "Polynomial Degree = 4", 
  '8' = "Polynomial Degree = 8",
  '12' = "Polynomial Degree = 12",
  '16' = "Polynomial Degree = 16",
  '20' = "Polynomial Degree = 20"
)
half_width_labeller <- as_labeller(half_width_names)
polynomial_degree_labeller <- as_labeller(polynomial_degree)

#SPO-Loss vs polynomial degree
spoplot <- results_relevant_fixed %>%
  ggplot(aes(x = as.factor(polykernel_degree), y = spo, fill = method)) +
  geom_boxplot() +
  scale_y_continuous(name = "Normalized SPO Loss", labels = scales::percent_format(accuracy = 1)) +
  scale_fill_discrete(name = "Method") +
  facet_wrap(vars(polykernel_noise_half_width), 
             labeller = labeller(polykernel_noise_half_width = half_width_labeller), 
             ncol = 2, scales = "free") + 
  theme_bw() +
  labs(x = "Polynomial Degree", title = "Normalized SPO Loss vs. Polynomial Degree") +
  theme(axis.title=element_text(size=36), axis.text=element_text(size=30), legend.text=element_text(size=36), 
        legend.title=element_text(size=36), strip.text = element_text(size = 24), 
        legend.position="top", plot.title = element_text(size = 42, hjust = 0.5))

ggsave("higher_pol_plot.png", width = 20, height = 18, units = "in")

####### R-SQUARED plots ######

#read in files: shortest_path_100n_5p, shortest_path_100n_10p, shortest_path_100n_20p
results <- rbind(shortest_path_100n_5p, shortest_path_100n_10p, shortest_path_100n_20p) 

#do for all polynomial degrees
results = results %>% filter(polykernel_degree == 16 & polykernel_noise_half_width == 0)

#calculate nomralized SPO+
results <- results %>% mutate(SPOplus_norm_spo = SPOplus_spoloss_test/zstar_avg_test,
                              LS_norm_spo = LS_spoloss_test/zstar_avg_test,
                              RF_norm_spo = RF_spoloss_test/zstar_avg_test,
                              Absolute_norm_spo = Absolute_spoloss_test/zstar_avg_test)

#rename
colnames(results)[colnames(results)=="SPOplus_norm_spo"] <- "SPO+"
colnames(results)[colnames(results)=="LS_norm_spo"] <- "Least Squares"
colnames(results)[colnames(results)=="RF_norm_spo"] <- "Random Forests"
colnames(results)[colnames(results)=="Absolute_norm_spo"] <- "Absolute Loss"

#keep relevant
results_relevant = results %>% 
  select(grid_dim, p_features, n_train, polykernel_degree, polykernel_noise_half_width,
         `SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`, `r_squared_adj`)

results_relevant_fixed = results_relevant %>%
  gather(`SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`,
         key = "method", value = "spo")

#SPO-Loss vs # adj_r2

p<-ggplot(results_relevant_fixed, aes(x=r_squared_adj, y=spo, group=method)) +
  geom_line(aes(color=method))+
  geom_point(aes(color=method)) +
  scale_y_continuous(name = "SPO Loss", labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Adjusted R-squared", title = "SPO Loss against Adjusted R-squared (degree = 4)") 
p

ggsave("r_square_pol4.png",  units = "in")


####### AGGREGATE DATA FOR FEATURE COMPARISON TABLE #########
results <- rbind(shortest_path_100n_5p, shortest_path_100n_10p, shortest_path_100n_20p) 
results <- results %>% mutate(SPOplus_norm_spo = SPOplus_spoloss_test/zstar_avg_test,
                              LS_norm_spo = LS_spoloss_test/zstar_avg_test,
                              RF_norm_spo = RF_spoloss_test/zstar_avg_test,
                              Absolute_norm_spo = Absolute_spoloss_test/zstar_avg_test)
#Selecting polynomial degrees
results1 = results %>% filter(polykernel_degree == 1 & polykernel_noise_half_width == 0.5)

#table with mean performance across different number of features' values
aggregate_results1 <- aggregate(result1[, c('SPOplus_norm_spo', 'LS_norm_spo',
                                            'RF_norm_spo', 'Absolute_norm_spo')], 
                                list(results[, c('p_features')]), mean)

results4 = results %>% filter(polykernel_degree == 4 & polykernel_noise_half_width == 0.5)

#table with mean performance across different number of features' values
aggregate_results4 <- aggregate(result4[, c('SPOplus_norm_spo', 'LS_norm_spo',
                                            'RF_norm_spo', 'Absolute_norm_spo')], 
                                list(results[, c('p_features')]), mean)
results8 = results %>% filter(polykernel_degree == 8 & polykernel_noise_half_width == 0.5)

#table with mean performance across different number of features' values
aggregate_results8 <- aggregate(result8[, c('SPOplus_norm_spo', 'LS_norm_spo',
                                            'RF_norm_spo', 'Absolute_norm_spo')], 
                                list(results[, c('p_features')]), mean)


######## SPO vs FEATURES BOXPLOT ########
#gather results
results <- rbind(shortest_path_100n_5p, shortest_path_100n_10p, shortest_path_100n_20p)
#keep relevant
results = results %>% filter(polykernel_noise_half_width == 0.5 & polykernel_degree <= 8)

#calculate normalized SPO+
results <- results %>% mutate(SPOplus_norm_spo = SPOplus_spoloss_test/zstar_avg_test,
                              LS_norm_spo = LS_spoloss_test/zstar_avg_test,
                              RF_norm_spo = RF_spoloss_test/zstar_avg_test,
                              Absolute_norm_spo = Absolute_spoloss_test/zstar_avg_test)


#rename
colnames(results)[colnames(results)=="SPOplus_norm_spo"] <- "SPO+"
colnames(results)[colnames(results)=="LS_norm_spo"] <- "Least Squares"
colnames(results)[colnames(results)=="RF_norm_spo"] <- "Random Forests"
colnames(results)[colnames(results)=="Absolute_norm_spo"] <- "Absolute Loss"

#keep relevant columns
results_relevant = results %>% 
  select(grid_dim, p_features, n_train, polykernel_degree, polykernel_noise_half_width,
         `SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`)

results_relevant_fixed = results_relevant %>%
  gather(`SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`,
         key = "method", value = "spo")

results_relevant_fixed$method = as.factor(results_relevant_fixed$method)
results_relevant_fixed$p_features = as.factor(results_relevant_fixed$p_features)


# Labelers
p_features_names <- c(
  '5' = "p = 5",
  '10' = "p = 10",
  '20' = "p = 20"
)

polynomial_degree <- c(
  '1' = "Polynomial Degree = 1",
  '4' = "Polynomial Degree = 4", 
  '8' = "Polynomial Degree = 8"
)

half_width_labeller <- as_labeller(half_width_names)
p_features_labeller <- as_labeller(p_features_names)
polynomial_degree_labeller <- as_labeller(polynomial_degree)


#SPO-Loss vs # Features
spoplot <- results_relevant_fixed %>%
  ggplot(aes(x = as.factor(p_features), y = spo, fill = method)) +
  geom_boxplot() +
  scale_y_continuous(name = "SPO Loss", labels = scales::percent_format(accuracy = 1)) +
  scale_fill_discrete(name = "Method") +
  facet_wrap(vars(polykernel_degree), 
             labeller = labeller(polykernel_degree = polynomial_degree_labeller), 
             ncol = 3, scales = "free") + 
  theme_bw() +
  labs(x = "Number of Features", title = "SPO Loss vs. Number of Features (Noise = 0.5)") +
  theme(axis.title=element_text(size=36), axis.text=element_text(size=30), legend.text=element_text(size=36), 
        legend.title=element_text(size=36), strip.text = element_text(size = 24), 
        legend.position="top", plot.title = element_text(size = 42, hjust = 0.5))

spoplot 

ggsave("features_boxplot_plot.png", width = 20, height = 18, units = "in")

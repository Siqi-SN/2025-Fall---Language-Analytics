Xgboost
================
Siqi
2025-12-6

``` r
rm(list = ls(all.names = TRUE))
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.4
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.0
    ## ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(tidymodels)
```

    ## ── Attaching packages ────────────────────────────────────── tidymodels 1.2.0 ──
    ## ✔ broom        1.0.5     ✔ rsample      1.2.1
    ## ✔ dials        1.3.0     ✔ tune         1.2.1
    ## ✔ infer        1.0.7     ✔ workflows    1.1.4
    ## ✔ modeldata    1.4.0     ✔ workflowsets 1.1.0
    ## ✔ parsnip      1.3.2     ✔ yardstick    1.3.1
    ## ✔ recipes      1.2.1     
    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## ✖ scales::discard() masks purrr::discard()
    ## ✖ dplyr::filter()   masks stats::filter()
    ## ✖ recipes::fixed()  masks stringr::fixed()
    ## ✖ dplyr::lag()      masks stats::lag()
    ## ✖ yardstick::spec() masks readr::spec()
    ## ✖ recipes::step()   masks stats::step()
    ## • Use suppressPackageStartupMessages() to eliminate package startup messages

``` r
library(janitor)
```

    ## 
    ## 载入程辑包：'janitor'
    ## 
    ## The following objects are masked from 'package:stats':
    ## 
    ##     chisq.test, fisher.test

``` r
library(xgboost)
```

    ## 
    ## 载入程辑包：'xgboost'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

``` r
library(vip)
```

    ## 
    ## 载入程辑包：'vip'
    ## 
    ## The following object is masked from 'package:utils':
    ## 
    ##     vi

``` r
library(rpart)
```

    ## 
    ## 载入程辑包：'rpart'
    ## 
    ## The following object is masked from 'package:dials':
    ## 
    ##     prune

``` r
library(rpart.plot)
```

``` r
ou <- read_csv("C:/Users/ASUS/Desktop/NLP/NLP Final project/final_project coursera_reviews.csv") %>%
  mutate(result = fct_relevel(as_factor(label), c("POS", "NEG")))
```

    ## Rows: 19461 Columns: 15
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (5): reviews, reviewers, date_reviews, course_id, label
    ## dbl (10): rating, word_count, positive_count, negative_count, ADJ_POS, ADV_P...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

# ——————

# Data preprocessing

``` r
ou <- ou %>%
  select(result,ADJ_POS,ADV_POS,VERB_POS,ADJ_NEG,ADV_NEG,VERB_NEG)
```

# train-test split

``` r
ou_split<-initial_split(ou)

ou_train<-training(ou_split)

ou_test<-testing(ou_split)
```

# XGBoost

## Formula and Recipe

``` r
rf_formula<-as.formula("result~.")
```

``` r
ou_rec<-recipe(rf_formula,ou_train)%>%
  update_role(result,new_role = "outcome")%>%
  step_corr(all_predictors(), threshold = 0.9)
```

## XGboost Specification

From: <https://juliasilge.com/blog/xgboost-tune-volleyball/>

``` r
xgb_spec <- boost_tree(
  trees = 100, 
  tree_depth = tune(), 
  min_n = tune(),
  loss_reduction = tune(), ## first three: model complexity
  sample_size = tune(), 
  mtry = tune(),         ## randomness
  learn_rate = tune() ## step size
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
```

``` r
xgb_grid <- grid_space_filling(
  tree_depth(),
  min_n(),
  loss_reduction(), 
  sample_size = sample_prop(),
  finalize(mtry(), ou_train),
  learn_rate(),
  size = 30
)
```

``` r
ou_wf <- workflow() %>%
  add_recipe(ou_rec) %>%
  add_model(xgb_spec)
```

# resampling inside (monte carlo) and outside (k-fold) the data

``` r
ou_rs<-ou%>%vfold_cv()
```

## Fit Model

``` r
fit_model<-TRUE

if(fit_model){
xg_tune_res <- tune_grid(
  ou_wf,
  grid=xgb_grid,
  resamples = ou_rs,
)
save(xg_tune_res,file="xg_tune_res.Rdata")

} else{
  load("xg_tune_res.Rdata")
}
```

``` r
xg_tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, mtry:sample_size) %>%
  pivot_longer(mtry:sample_size,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")
```

![](11-xgboost_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
show_best(xg_tune_res,metric="roc_auc")
```

    ## # A tibble: 5 × 12
    ##    mtry min_n tree_depth learn_rate loss_reduction sample_size .metric
    ##   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>  
    ## 1     6     9          9  0.1        0.00000230          0.814 roc_auc
    ## 2     2    24          4  0.0489     0.329               0.783 roc_auc
    ## 3     2     8         11  0.0000386  0.0000000001        0.659 roc_auc
    ## 4     4    33         13  0.0240     0.00000000961       0.628 roc_auc
    ## 5     2    13         15  0.000672   0.0213              0.752 roc_auc
    ## # ℹ 5 more variables: .estimator <chr>, mean <dbl>, n <int>, std_err <dbl>,
    ## #   .config <chr>

``` r
best_auc <- select_best(xg_tune_res,metric =  "roc_auc")
best_auc
```

    ## # A tibble: 1 × 7
    ##    mtry min_n tree_depth learn_rate loss_reduction sample_size .config          
    ##   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>            
    ## 1     6     9          9        0.1     0.00000230       0.814 Preprocessor1_Mo…

``` r
final_xgb <- finalize_workflow(
  ou_wf,
  best_auc
)
```

## Variable importance

How xgb.importance() Computes and Scales Feature Importance

``` r
final_xgb %>%
  fit(data = ou_train) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point")+
  geom_point(color="blue")+
  theme_minimal()
```

![](11-xgboost_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
final_xgb %>%
  fit(data = ou_train) %>%
  extract_fit_parsnip() %>%
  vip(geom = "col") +   # 改成蓝色柱状图
  theme_minimal()
```

![](11-xgboost_files/figure-gfm/unnamed-chunk-16-2.png)<!-- -->

## Metrics

``` r
final_res <- last_fit(final_xgb, ou_split)

collect_metrics(final_res)
```

    ## # A tibble: 3 × 4
    ##   .metric     .estimator .estimate .config             
    ##   <chr>       <chr>          <dbl> <chr>               
    ## 1 accuracy    binary         0.849 Preprocessor1_Model1
    ## 2 roc_auc     binary         0.926 Preprocessor1_Model1
    ## 3 brier_class binary         0.108 Preprocessor1_Model1

pkgs <- list("data.table", "lubridate", "dplyr",
    "ggplot2", "caret", "partykit", "plyr",
    "C50", "glmnet", "doParallel", "foreach", "pROC")
lapply(pkgs, require, character.only = T)

my_chart_color <- "#FF7600"
my_char_attributes <- theme(
    text = element_text(family = "Helvetica"),
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "gray"),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank())


f_na <- function(DT) {
    for (j in seq_len(ncol(DT))) {
        set(DT, which(is.na(DT[[j]])), j, 0)
    }
    DT
}

data <- fread("Final_data_cleaned.csv")

# RENAME TASK RUNS THAT LAST LESS THAN 10 SECONDS
data <- data[task_time >= 10 & task_time < 900, ]

data <- data[, logpenaltydel := log(penalty_del + 0.01)]
data <- data[, logpenaltynav := log(penalty_nav + 0.01)]
demographics <- fread("demographics.csv")
data <- merge(data, demographics, by = "user_id")
commands <- fread("commands.csv")
del_comm_list <- commands[grepl("Delete", reduces), command_id]
nav_comm_list <- commands[grepl("Navigation", reduces), command_id]
nav_comm_list <- nav_comm_list[nav_comm_list %in% names(data)]
data[, del_commands := rowSums(.SD, na.rm = TRUE),
    .SDcols = del_comm_list]
data[, nav_commands := rowSums(.SD, na.rm = TRUE),
    .SDcols = nav_comm_list]

data[, is_in_task_del := ifelse(penalty_del != 0 | del_commands != 0, 1, 0)]
data[, is_in_task_nav := ifelse(penalty_nav != 0 | nav_commands != 0, 1, 0)]

write.csv(data, "Final_data_cleaned_agg.csv")

vars <- c("user_id", "editor", "task_runs", "first_time",
    "penalty_del", "penalty_nav", "task_time", "del_commands",
    "is_in_task_del", "is_in_task_nav", "logpenaltydel",
    "logpenaltynav", "nav_commands", names(demographics)[-1])
data <- data[, vars, with = F]
data <- f_na(data)
data[, task_ocurrence := as.POSIXct(first_time,
    format = "%Y-%m-%dT%H:%M:%OS", tz = "GMT")]
data[, task_ocurrence := format(task_ocurrence,
    usetz = TRUE, tz = "America/Bogota")]
data[, task_day := day(task_ocurrence)]
data[, task_month := month(task_ocurrence)]
data[, task_year := year(task_ocurrence)]
data[, task_dayweek := weekdays(as.Date(task_ocurrence))]
data[, used_command_del := ifelse(del_commands > 0, 1, 0)]
data[, used_command_nav := ifelse(nav_commands > 0, 1, 0)]
data[, morning_after := ifelse(hour(task_ocurrence) < 12,
    "Morning", "Afternoon")]
data_del <- data %>% filter(is_in_task_del == 1)
data_nav <- data %>% filter(is_in_task_nav == 1)
write.csv(data_del, "model_features_del.csv")
write.csv(data_nav, "model_features_nav.csv")

data_all <- merge(data_del[, c("user_id", "editor", "task_runs")],
    data_nav[, 
        c("user_id", "editor", "task_runs", "penalty_del", "penalty_nav")],
    by = c("user_id","editor", "task_runs"))

daily_data <- data[, lapply(.SD, sum, na.rm = TRUE),
    by = .(user_id, editor, task_year, task_month, task_day, task_dayweek),
    .SDcols = c("del_commands", "nav_commands", "penalty_del",
        "penalty_nav", "task_time")]
png("good_vs_bad_del_all.png", width = 800, height = 600)
data_del %>%
    ggplot(aes(x = del_commands, y = penalty_del)) +
    geom_point(color = my_chart_color) +
    labs(x = "Number of shortcut commands associated with delete tasks",
        y = "Duration of delete penalty (Seconds)",
        caption = "Source: Kognos Software | CADMS"
        , title = paste0("Behavior of the penalty times for delete tasks",
            " compared to the number\n of shortcut command",
            " usage by task run"),
        subtitle = paste0("The penalty times for all task runs are",
            " shown separated by the number of shortcut commands",
            " associated with deletes.\n We can observe that as the",
            " number of shortcut commands used increases, the ",
            "probability\n of observing high penalty times",
            " decreases")
        ) +
    theme_bw() +
    my_char_attributes
dev.off()

png("good_vs_bad_nav_all.png", width = 800, height = 600)
data_nav %>%
    ggplot(aes(x = nav_commands, y = penalty_nav)) +
    geom_point(color = my_chart_color) +
    labs(x = "Number of shortcut commands associated with navigation tasks",
        y = "Duration of navigation penalty (Seconds)",
        caption = "Source: Kognos Software | CADMS"
         ,title = paste0("Behavior of the penalty times for navigation",
             " tasks compared to the number\n of shortcut command",
             " usage by task run"),
         subtitle = paste0("The penalty times for all task runs are",
             " shown separated by the number of shortcut commands",
             " associated with navigation.\n We can observe that as",
             " the number of shortcut commands used increases, the ",
             "probability\n of observing high penalty times",
             " decreases"
             )
        ) +
    theme_bw() +
    my_char_attributes
dev.off()

indep_del <- c("task_time", "del_commands", "task_dayweek",
    "editor", "user_id", "company", "degree", "years_study",
    "penalty_nav", "work_experience", "morning_after")
dep_del <- "penalty_del"

indep_nav <- c("task_time", "nav_commands", "task_dayweek",
    "editor", "user_id", "company", "degree", "years_study",
    "penalty_del", "work_experience", "morning_after")
dep_nav <- "penalty_nav"


# Model bench marking
# (See Hothorn at al, "The design and analysis of benchmark experiments
# -Journal of Computational and Graphical Statistics (2005) vol 14 (3) 
# pp 675-699) 

registerDoParallel(cores = 4)
companies <- unique(data$company)
users <- unique(data$user_id)
editors <- unique(data$editor)
train_del <- data.frame(matrix(NA, nrow = 1, ncol = ncol(data_del)))
names(train_del) <- names(data_del)
train_nav <- data.frame(matrix(NA, nrow = 1, ncol = ncol(data_nav)))
names(train_nav) <- names(data_nav)


# is paired sample for all models since we set seed
set.seed(2018)
# Stratified sampling
for (company_iter in companies){
    for (user_iter in users) {
        for (editor_iter in editors){
            data_smp_del <- data_del %>% 
                filter(company == company_iter &
                    editor == editor_iter &
                    user_id == user_iter)
            if(nrow(data_smp_del) > 0) {
                data_smp_del <- data_smp_del %>% sample_frac(0.5)
                train_del <- rbind(train_del, data_smp_del)
            }
            data_smp_nav <- data_nav %>% 
                filter(company == company_iter &
                    editor == editor_iter &
                    user_id == user_iter)
            if(nrow(data_smp_nav) > 0) {
                data_smp_nav <- data_smp_nav %>% sample_frac(0.5)
                train_nav <- rbind(train_nav, data_smp_nav)
            }
        }
    }
}

train_del <- train_del[-1, ]
train_nav <- train_nav[-1, ]
test_del <- anti_join(data_del,
    train_del, by = c("company", "user_id", "editor", "task_runs"))
test_nav <- anti_join(data_nav,
    train_nav, by = c("company", "user_id", "editor", "task_runs"))
test_x_del <- test_del[, indep_del]
test_y_del <- test_del$penalty_del
test_x_nav <- test_nav[, indep_nav]
test_y_nav <- test_nav$penalty_nav

# 10-fold CV with 5 repeats for all models
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# Conditional inference Tree

# Deletes

fit_ctree_del <- train(penalty_del ~ task_time + del_commands
    + task_dayweek + editor  +
    factor(user_id) + company + degree + years_study + work_experience +
    morning_after,
    data = train_del, method = "ctree",
    tuneGrid = expand.grid(mincriterion = c(seq(0.1, 0.95, by = 0.05))),
    trControl = fitControl)

ctree_test_del <- predict(fit_ctree_del, test_x_del)
tree_model_del <- fit_ctree_del$best_model

ctreeVarImp_del <- varImp(fit_ctree_del)
ctreeVarImp_del

png("ctree_varimp_del.png", width = 800, height = 600)
plot(ctreeVarImp_del)
dev.off()

# Navigation

fit_ctree_nav <- train(penalty_nav ~ task_time + nav_commands
    + task_dayweek + editor  +
    factor(user_id) + company + degree + years_study + work_experience +
    morning_after,
    data = train_nav, method = "ctree",
    tuneGrid = expand.grid(mincriterion = c(seq(0.1, 0.95, by = 0.05))),
    trControl = fitControl)

ctree_test_nav <- predict(fit_ctree_nav, test_x_nav)
tree_model_nav <- fit_ctree_nav$best_model

ctreeVarImp_nav <- varImp(fit_ctree_nav)
ctreeVarImp_nav

png("ctree_varimp_nav.png", width = 800, height = 600)
plot(ctreeVarImp_nav)
dev.off()


# Elastic Nets (Generalized model that combines lasso and ridge regression)
fit_enet_del <- train(round(penalty_del, 0) ~ task_time + del_commands
    + task_dayweek + editor  +
    factor(user_id) + company + degree +
    years_study + work_experience +
    morning_after,
    data = train_del, method = "glmnet",
    family = "poisson",
    trControl = fitControl)

glmnet_model_del <- fit_enet_del$finalModel
enet_test_del <- predict(fit_enet_del, test_x_del)
print(coef(glmnet_model_del, s = fit_enet_del$bestTune$lambda))
enetVarImp_del <- varImp(fit_enet_del)
enetVarImp_del

png("glmnet_varimp_del.png", width = 800, height = 600)
plot(enetVarImp_del)
dev.off()

fit_enet_nav <- train(round(penalty_nav, 0) ~ task_time + nav_commands
    + task_dayweek + editor  +
    factor(user_id) + company + degree + years_study +
    work_experience + morning_after,
    data = train_nav, method = "glmnet",
    family = "poisson",
    trControl = fitControl)

glmnet_model_nav <- fit_enet_nav$finalModel
enet_test_nav <- predict(fit_enet_nav, test_x_nav)
print(coef(glmnet_model_nav, s = fit_enet_nav$bestTune$lambda))
enetVarImp_nav <- varImp(fit_enet_nav)
enetVarImp_nav

png("glmnet_varimp_nav.png", width = 800, height = 600)
plot(enetVarImp_nav)
dev.off()

# Extreme gradient boosting
fit_xgb_del <- train(penalty_del ~ task_time + del_commands
    + task_dayweek + editor  +
    factor(user_id) + company + degree + years_study + work_experience +
    morning_after,
    data = train_del, method = "xgbTree",
    trControl = fitControl)

xgb_model_del <- fit_xgb_del$finalModel
xgb_test_del <- predict(fit_xgb_del, test_x_del)

xgbVarImp_del <- varImp(fit_xgb_del)

png("xgb_varimp_del.png", width = 800, height = 600)
plot(xgbVarImp_del)
dev.off()


fit_xgb_nav <- train(penalty_nav ~ task_time + nav_commands
    + task_dayweek + editor  +
    factor(user_id) + company + degree + years_study + work_experience +
    morning_after,
    data = train_nav, method = "xgbTree",
    trControl = fitControl)

xgb_monav_nav <- fit_xgb_nav$finalModel
xgb_test_nav <- predict(fit_xgb_nav, test_x_nav)

xgbVarImp_nav <- varImp(fit_xgb_nav)

xgbVarImp_nav
png("xgb_varimp_nav.png", width = 800, height = 600)
plot(xgbVarImp_nav)
dev.off()

bench <- data.frame(
    postResample(ctree_test_del, test_y_del), 
    postResample(enet_test_del, test_y_del),
    postResample(xgb_test_del, test_y_del))

names(bench) <- c("ctree", "glmnet", "xgboost")

bench <- data.frame(
    postResample(ctree_test_nav, test_y_nav), 
    postResample(enet_test_nav, test_y_nav), 
    postResample(xgb_test_nav, test_y_nav) 
    )

names(bench) <- c("ctree", "glmnet", "xgboost")

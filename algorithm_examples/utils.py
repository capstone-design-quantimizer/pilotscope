import os
import json
from datetime import datetime
from pathlib import Path

from pilotscope.Common.Index import Index
from pilotscope.Common.Util import json_str_to_json_obj
from pilotscope.Common.TimeStatistic import TimeStatistic
from pilotscope.Common.Drawer import Drawer
from pilotscope.DBController.BaseDBController import BaseDBController
from pilotscope.Dataset.ImdbDataset import ImdbDataset
from pilotscope.Dataset.StatsDataset import StatsDataset
from pilotscope.Dataset.StatsTinyDataset import StatsTinyDataset
from pilotscope.Dataset.TpcdsDataset import TpcdsDataset
from pilotscope.PilotEnum import DatabaseEnum


def get_path(sql_file):
    my_path = os.path.abspath(__file__)
    return os.path.join(os.path.dirname(my_path), sql_file)


def load_training_sql(db):
    if "stats_tiny" == db.lower():
        return StatsTinyDataset(DatabaseEnum.POSTGRESQL).read_train_sql()
    elif "stats" in db.lower():
        return StatsDataset(DatabaseEnum.POSTGRESQL).read_train_sql()
    elif "imdb" in db:
        return ImdbDataset(DatabaseEnum.POSTGRESQL).read_train_sql()
    elif "tpcds" in db.lower():
        return TpcdsDataset(DatabaseEnum).read_train_sql()
    else:
        raise NotImplementedError


def load_test_sql(db):
    if "stats_tiny" == db.lower():
        return StatsTinyDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    elif "stats" in db.lower():
        return StatsDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    elif "imdb" in db:
        return ImdbDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    elif "tpcds" in db.lower():
        return TpcdsDataset(DatabaseEnum).read_test_sql()
    else:
        raise NotImplementedError


def load_sql(file):
    with open(file) as f:
        sqls = []
        line = f.readline()
        while line is not None and line != "":
            if "#" in line:
                sqls.append(line.split("#####")[1])
            else:
                sqls.append(line)
            line = f.readline()
        return sqls


def scale_card(subquery_2_card: dict, factor):
    res = {}
    for key, value in subquery_2_card.items():
        res[key] = value * factor
    return res


def compress_anchor_name(name_2_values):
    res = {}
    for name, value in name_2_values.items():
        res[name.split("_")[0]] = value
    return res


def recover_stats_index(db_controller: BaseDBController):
    db_controller.drop_all_indexes()
    db_controller.execute("create index idx_posts_owneruserid on posts using btree(owneruserid);")
    db_controller.execute("create index  idx_posts_lasteditoruserid on posts using btree(lasteditoruserid);")
    db_controller.execute("create index idx_postlinks_relatedpostid on postLinks using btree(relatedpostid);")
    db_controller.execute("create index idx_postlinks_postid on postLinks using btree(postid);")
    db_controller.execute("create index idx_posthistory_postid on postHistory using btree(postid);")
    db_controller.execute("create index idx_posthistory_userid on postHistory using btree(userid);")
    db_controller.execute("create index idx_comments_postid on comments using btree(postid);")
    db_controller.execute("create index idx_comments_userid on comments using btree(userid);")
    db_controller.execute("create index idx_votes_userid on votes using btree(userid);")
    db_controller.execute("create index idx_votes_postid on votes using btree(postid);")
    db_controller.execute("create index idx_badges_userid on badges using btree(userid);")
    db_controller.execute("create index idx_tags_excerptpostid on tags using btree(excerptpostid);")


def recover_imdb_index(db_controller: BaseDBController):
    queries = [
        'CREATE INDEX "person_id_aka_name" ON "public"."aka_name" USING btree ("person_id");',
        'CREATE INDEX "kind_id_aka_title" ON "public"."aka_title" USING btree ("kind_id");',
        'CREATE INDEX "movie_id_aka_title" ON "public"."aka_title" USING btree ("movie_id");',
        'CREATE INDEX "movie_id_cast_info" ON "public"."cast_info" USING btree ("movie_id");',
        'CREATE INDEX "person_id_cast_info" ON "public"."cast_info" USING btree ("person_id");',
        'CREATE INDEX "person_role_id_cast_info" ON "public"."cast_info" USING btree ("person_role_id");',
        'CREATE INDEX "role_id_cast_info" ON "public"."cast_info" USING btree ("role_id");',
        'CREATE INDEX "movie_id_complete_cast" ON "public"."complete_cast" USING btree ("movie_id");',
        'CREATE INDEX "status_id_complete_cast" ON "public"."complete_cast" USING btree ("status_id");',
        'CREATE INDEX "subject_id_complete_cast" ON "public"."complete_cast" USING btree ("subject_id");',
        'CREATE INDEX "company_id_movie_companies" ON "public"."movie_companies" USING btree ("company_id");',
        'CREATE INDEX "company_type_id_movie_companies" ON "public"."movie_companies" USING btree ("company_type_id");',
        'CREATE INDEX "movie_id_movie_companies" ON "public"."movie_companies" USING btree ("movie_id");',
        'CREATE INDEX "info_type_id_movie_info" ON "public"."movie_info" USING btree ("info_type_id");',
        'CREATE INDEX "movie_id_movie_info" ON "public"."movie_info" USING btree ("movie_id");',
        'CREATE INDEX "info_type_id_movie_info_idx" ON "public"."movie_info_idx" USING btree ("info_type_id");',
        'CREATE INDEX "movie_id_movie_info_idx" ON "public"."movie_info_idx" USING btree ("movie_id");',
        'CREATE INDEX "keyword_id_movie_keyword" ON "public"."movie_keyword" USING btree ("keyword_id");',
        'CREATE INDEX "movie_id_movie_keyword" ON "public"."movie_keyword" USING btree ("movie_id");',
        'CREATE INDEX "link_type_id_movie_link" ON "public"."movie_link" USING btree ("link_type_id");',
        'CREATE INDEX "linked_movie_id_movie_link" ON "public"."movie_link" USING btree ("linked_movie_id");',
        'CREATE INDEX "movie_id_movie_link" ON "public"."movie_link" USING btree ("movie_id");',
        'CREATE INDEX "info_type_id_person_info" ON "public"."person_info" USING btree ("info_type_id");',
        'CREATE INDEX "person_id_person_info" ON "public"."person_info" USING btree ("person_id");',
        'CREATE INDEX "kind_id_title" ON "public"."title" USING btree ("kind_id");'
    ]

    for query in queries:
        db_controller.execute(query)


def to_pilot_index(index):
    columns = [c.name for c in index.columns]
    pilot_index = Index(columns=columns, table=index.table().name, index_name=index.index_idx())
    if hasattr(index, "hypopg_oid"):
        pilot_index.hypopg_oid = index.hypopg_oid
    if hasattr(index, "hypopg_name"):
        pilot_index.hypopg_name = index.hypopg_name
    return pilot_index


def to_tree_json(spark_plan):
    plan = json_str_to_json_obj(spark_plan)
    if "Plan" in plan and isinstance(plan["Plan"], list):
        plan["Plan"], _ = _to_tree_json(plan["Plan"], 0)
    else:
        plan["Plan"], _ = _to_tree_json(plan["inputPlan"], 0)
    return plan


def _to_tree_json(targets, index=0):
    node = targets[index]
    num_children = node["num-children"]

    all_child_node_size = 0
    if num_children == 0:
        # +1 is self
        return node, all_child_node_size + 1

    left_node, left_size = _to_tree_json(targets, index + all_child_node_size + 1)
    node["Plans"] = [left_node]
    all_child_node_size += left_size

    if num_children == 2:
        right_node, right_size = _to_tree_json(targets, index + all_child_node_size + 1)
        node["Plans"].append(right_node)
        all_child_node_size += right_size

    return node, all_child_node_size + 1


def get_spark_table_name_for_scan_node(node: dict):
    node_type = node["class"]
    if "org.apache.spark.sql.execution.columnar.InMemoryTableScanExec" == node_type:
        table = node["relation"][0]["cacheBuilder"]["tableName"]
        assert len(node["relation"]) == 1
    elif "org.apache.spark.sql.execution.RowDataSourceScanExec" == node_type:
        table = node["output"][0][0]["name"]
        assert len(node["output"][0]) == 1
    else:
        raise NotImplementedError
    return table


# ================== Test Result Management Utils ==================

def save_test_result(algo_name, db_name, extra_info=None):
    """
    Save test results in JSON format for later comparison.
    Call this at the end of each test to persist results.
    
    Args:
        algo_name: Algorithm name (e.g., 'baseline', 'mscn', 'lero')
        db_name: Database name (e.g., 'stats_tiny', 'imdb')
        extra_info: Optional dict with additional metadata
    
    Returns:
        Path to the saved JSON file
    """
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{algo_name}_{db_name}_{timestamp}"
    
    # Save as JSON for easy loading and comparison
    json_data = {
        "algorithm": algo_name,
        "database": db_name,
        "timestamp": timestamp,
        "metrics": {
            "total_time": TimeStatistic.get_sum_data(),
            "average_time": TimeStatistic.get_average_data(),
            "query_count": TimeStatistic.get_count_data()
        }
    }
    
    if extra_info:
        json_data["extra_info"] = extra_info
    
    json_path = results_dir / f"{base_filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"📄 Results saved: {json_path}")
    
    return json_path


def load_test_results(result_files):
    """
    Load multiple test result JSON files for comparison.
    
    Args:
        result_files: List of JSON file paths (strings or Path objects)
    
    Returns:
        Dictionary mapping algorithm names to their metrics
    """
    results = {}
    
    for file_path in result_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        algo_name = data['algorithm']
        results[algo_name] = data
    
    return results


def compare_algorithms(result_files, metric='total_time', output_path=None, use_db_time=True):
    """
    Compare multiple algorithm test results and generate a comparison chart.
    
    Args:
        result_files: List of JSON file paths to compare
        metric: Which metric to compare ('total_time', 'average_time', 'query_count')
        output_path: Optional path to save the comparison chart (without extension)
        use_db_time: If True, use db_execution_time for fair comparison (default: True)
    
    Example:
        compare_algorithms([
            'results/baseline_stats_tiny_20231014_120000.json',
            'results/mscn_stats_tiny_20231014_130000.json',
            'results/lero_stats_tiny_20231014_140000.json'
        ], metric='total_time', output_path='results/comparison')
    """
    results = load_test_results(result_files)
    
    # Extract the specified metric from each result
    comparison_data = {}
    for algo_name, data in results.items():
        # 순수 DB 실행 시간 사용 (AI 추론 시간 제외)
        if use_db_time and 'extra_info' in data and 'db_execution_time' in data['extra_info']:
            db_time = data['extra_info']['db_execution_time']
            if db_time > 0:
                comparison_data[algo_name] = db_time
                continue
        
        # Fallback: 기존 방식 (TimeStatistic에서 가져온 시간)
        if metric in data['metrics']:
            metric_values = data['metrics'][metric]
            # If it's a dict, get the first value (usually there's only one key)
            if isinstance(metric_values, dict):
                comparison_data[algo_name] = sum(metric_values.values())
            else:
                comparison_data[algo_name] = metric_values
        else:
            print(f"⚠️  Metric '{metric}' not found in {algo_name} results")
    
    # Generate comparison chart
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/comparison_{metric}_{timestamp}"
    
    time_label = "DB Execution Time" if use_db_time else metric.replace('_', ' ').title()
    
    Drawer.draw_bar(
        comparison_data,
        output_path,
        x_title="Algorithms",
        y_title=f"{time_label} (s)",
        is_rotation=True
    )
    
    print(f"\n📊 Comparison chart saved: {output_path}.png")
    print(f"\n{'='*60}")
    print(f"Comparison Results ({time_label}):")
    print(f"{'='*60}")
    for algo, value in sorted(comparison_data.items(), key=lambda x: x[1]):
        print(f"  {algo:15s}: {value:10.4f}s")
    print(f"{'='*60}")
    
    return comparison_data


def list_saved_results(results_dir="results"):
    """
    List all saved test results grouped by algorithm and database.
    
    Returns:
        Dictionary mapping (algorithm, database) to list of result files
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"⚠️  Results directory '{results_dir}' not found")
        return {}
    
    grouped_results = {}
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            key = (data['algorithm'], data['database'])
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append({
                'file': str(json_file),
                'timestamp': data['timestamp'],
                'total_time': sum(data['metrics']['total_time'].values())
            })
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")
    
    # Sort by timestamp
    for key in grouped_results:
        grouped_results[key].sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Saved Test Results:")
    print(f"{'='*60}")
    for (algo, db), files in sorted(grouped_results.items()):
        print(f"\n{algo} on {db}: ({len(files)} results)")
        for result in files[:3]:  # Show latest 3
            print(f"  • {result['timestamp']} - {result['total_time']:.4f}s")
            print(f"    {result['file']}")
    print(f"{'='*60}\n")
    
    return grouped_results

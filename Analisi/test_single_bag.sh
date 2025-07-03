#!/bin/bash

# Script per processare tutte le rosbag con trasformazione statica
MAX_RATE_SPEEDUP="2.5"

echo "🚀 Script di processamento batch per tutte le rosbag"
echo "⚡ Velocità massima: ${MAX_RATE_SPEEDUP}x"
echo ""

# Trova tutte le cartelle rosbag nella directory corrente
ROSBAG_DIRS=($(find . -maxdepth 1 -type d -name "rosbag2_*" ! -name "*_processed" | sort))
# ROSBAG_DIRS=($(find /mnt/c/Users/valio/OneDrive\ -\ Università\ degli\ Studi\ di\ Perugia\ \(1\)/ETH_Valerio_Magnetico/ROSBAGS/ -maxdepth 1 -type d -name "*_processed" | sort))

if [ ${#ROSBAG_DIRS[@]} -eq 0 ]; then
    echo "❌ Nessuna rosbag trovata nella directory corrente"
    echo "🔍 Cerco cartelle con pattern: rosbag2_*"
    exit 1
fi

echo "📋 Rosbag trovate: ${#ROSBAG_DIRS[@]}"
for bag in "${ROSBAG_DIRS[@]}"; do
    echo "  📂 $(basename "$bag")"
done
echo ""

# Contatori per statistiche finali
TOTAL_PROCESSED=0
TOTAL_FAILED=0
PROCESSED_BAGS=()
FAILED_BAGS=()

# Funzione per processare una singola rosbag
process_rosbag() {
    local TARGET_BAG="$1"
    local BAG_NAME=$(basename "$TARGET_BAG")
    local OUTPUT_PATH="${TARGET_BAG}_processed"
    
    echo "🎬 =============================================="
    echo "🎬 Processando: $BAG_NAME"
    echo "🎬 =============================================="
    
    # Verifica che la rosbag esista
    if [ ! -d "$TARGET_BAG" ]; then
        echo "❌ Rosbag $TARGET_BAG non trovata"
        return 1
    fi
    
    # Salta se già processata
    if [ -d "$OUTPUT_PATH" ]; then
        echo "⚠️  Output già esistente, rimuovo: $OUTPUT_PATH"
        rm -rf "$OUTPUT_PATH"
    fi

    # 1. Pulizia processi precedenti
    echo "🧹 Step 1/8: Pulizia processi precedenti..."
    pkill -f "ros2 bag play" &>/dev/null || true
    pkill -f "static_transform_publisher" &>/dev/null || true
    pkill -f "ros2 bag record" &>/dev/null || true
    pkill -f "ekf_tf_republisher" &>/dev/null || true
    pkill -f "python3 -c" &>/dev/null || true
    sleep 2
    echo "✅ Pulizia completata"

    # 2. Avvio trasformazioni statiche
    echo "🔗 Step 2/8: Avvio trasformazioni statiche..."
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 vicon/robodog_magnetic/robodog_magnetic world 2>/dev/null &
    TRANSFORM_PID=$!
    echo "✅ Trasformazione statica world avviata (PID: $TRANSFORM_PID)"
    
    # Avvio anche il publisher della tf EKF_CF
    echo "🔗 Avvio publisher TF personalizzato per EKF_CF..."
    python3 -c "
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import sys

class EKFTFRepublisher(Node):
    def __init__(self):
        super().__init__('ekf_tf_republisher')
        
        # Buffer e listener per ricevere le tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Broadcaster per pubblicare la nuova tf
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timer per pubblicare a 50Hz
        self.timer = self.create_timer(0.02, self.republish_tf)
        
        self.get_logger().info('EKF TF Republisher avviato')
    
    def republish_tf(self):
        try:
            # Cerca la trasformazione cf21 -> world
            transform = self.tf_buffer.lookup_transform(
                'vicon/world',  # target frame
                'cf21',         # source frame
                rclpy.time.Time()  # latest available
            )
            
            # Crea una nuova trasformazione EKF_CF -> vicon/world
            ekf_transform = TransformStamped()
            ekf_transform.header.stamp = self.get_clock().now().to_msg()
            ekf_transform.header.frame_id = 'vicon/world'
            ekf_transform.child_frame_id = 'EKF_CF'
            
            # Copia la trasformazione
            ekf_transform.transform = transform.transform
            
            # Pubblica la nuova tf
            self.tf_broadcaster.sendTransform(ekf_transform)
            
        except Exception as e:
            # Non stampare errori per evitare spam
            pass

def main():
    rclpy.init()
    node = EKFTFRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
" 2>/dev/null &
    EKF_TF_PID=$!
    echo "✅ EKF TF Republisher avviato (PID: $EKF_TF_PID)"
    sleep 2

    # 3. Avvio registrazione rosbag
    echo "📹 Step 3/8: Avvio registrazione nuova rosbag..."
    ros2 bag record -a -o "$OUTPUT_PATH" 2>/dev/null &
    RECORD_PID=$!
    echo "✅ Registrazione avviata (PID: $RECORD_PID)"
    sleep 3

    # 4. Play rosbag originale
    echo "▶️  Step 4/8: Avvio playback rosbag originale..."
    ros2 bag play "$TARGET_BAG" --rate "$MAX_RATE_SPEEDUP" --delay 2 &
    PLAY_PID=$!
    echo "✅ Playback avviato (PID: $PLAY_PID)"

    # 5. Pausa per 2 secondi per discovery dei topic
    echo "⏸️  Step 5/8: Pausa per discovery topic (2 secondi)..."
    sleep 2
    echo "✅ Discovery completato"

    # 6. Continua il playback (già in esecuzione)
    echo "▶️  Step 6/8: Playback in corso..."
    wait $PLAY_PID 2>/dev/null || true
    echo "✅ Playback completato"

    # 7. Attesa finalizzazione
    echo "⏳ Step 7/8: Attesa finalizzazione (3 secondi)..."
    sleep 3
    echo "✅ Finalizzazione completata"

    # 8. Chiusura processi
    echo "⏹️  Step 8/8: Chiusura processi..."
    kill $RECORD_PID &>/dev/null || true
    kill $TRANSFORM_PID &>/dev/null || true
    kill $EKF_TF_PID &>/dev/null || true
    sleep 2
    pkill -f "ros2 bag play" &>/dev/null || true
    pkill -f "static_transform_publisher" &>/dev/null || true
    pkill -f "ros2 bag record" &>/dev/null || true
    pkill -f "ekf_tf_republisher" &>/dev/null || true
    pkill -f "python3 -c" &>/dev/null || true
    echo "✅ Tutti i processi chiusi"

    # Verifica risultato
    if [ -d "$OUTPUT_PATH" ]; then
        echo "✅ Rosbag processata con successo: $BAG_NAME"
        return 0
    else
        echo "❌ Errore nel processamento di: $BAG_NAME"
        return 1
    fi
}

# Funzione per analizzare una rosbag (versione semplificata)
analyze_rosbag() {
    local TARGET_BAG="$1"
    local OUTPUT_PATH="${TARGET_BAG}_processed"
    local BAG_NAME=$(basename "$TARGET_BAG")
    
    echo ""
    echo "� ANALISI: $BAG_NAME"
    echo "========================"
    
    # Funzione per estrarre info sui messaggi
    get_topic_messages() {
        local bag_path="$1"
        ros2 bag info "$bag_path" --verbose 2>/dev/null | grep "Topic:" | awk -F'|' '{
            gsub(/^[ \t]+|[ \t]+$/, "", $1); 
            gsub(/^[ \t]+|[ \t]+$/, "", $3); 
            gsub(/.*Topic: /, "", $1); 
            gsub(/.*Count: /, "", $3); 
            print $1":"$3
        }'
    }
    
    # Analizza rosbag originale
    declare -A original_topics
    while IFS=: read -r topic count; do
        if [[ -n "$topic" && -n "$count" && "$count" != "0" ]]; then
            original_topics["$topic"]="$count"
        fi
    done < <(get_topic_messages "$TARGET_BAG")
    
    # Analizza rosbag processata
    declare -A processed_topics
    while IFS=: read -r topic count; do
        if [[ -n "$topic" && -n "$count" && "$count" != "0" ]]; then
            processed_topics["$topic"]="$count"
        fi
    done < <(get_topic_messages "$OUTPUT_PATH")
    
    # Calcola totali
    local total_original=0
    local total_processed=0
    
    for count in "${original_topics[@]}"; do
        total_original=$((total_original + count))
    done
    
    for count in "${processed_topics[@]}"; do
        total_processed=$((total_processed + count))
    done
    
    # Trova topic aggiunti
    local added_topics=()
    for topic in "${!processed_topics[@]}"; do
        if [[ -z "${original_topics[$topic]:-}" ]]; then
            added_topics+=("$topic")
        fi
    done
    
    echo "📈 Messaggi originali: $total_original"
    echo "📈 Messaggi processati: $total_processed"
    echo "📈 Messaggi aggiunti: $((total_processed - total_original))"
    
    if [ ${#added_topics[@]} -gt 0 ]; then
        echo "➕ Topic aggiunti: ${#added_topics[@]}"
        for topic in "${added_topics[@]}"; do
            echo "   - $topic (${processed_topics[$topic]} msg)"
        done
    fi
    
    echo ""
}

# Loop principale per processare tutte le rosbag
echo "� Inizio processamento batch..."
echo ""

for bag_dir in "${ROSBAG_DIRS[@]}"; do
    BAG_NAME=$(basename "$bag_dir")
    
    echo "⏱️  Processando $((TOTAL_PROCESSED + TOTAL_FAILED + 1))/${#ROSBAG_DIRS[@]}: $BAG_NAME"
    
    if process_rosbag "$bag_dir"; then
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
        PROCESSED_BAGS+=("$BAG_NAME")
        analyze_rosbag "$bag_dir"
        echo "✅ $BAG_NAME processata con successo!"
    else
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_BAGS+=("$BAG_NAME")
        echo "❌ Errore nel processamento di $BAG_NAME"
    fi
    
    echo ""
    echo "💤 Pausa tra processamenti (3 secondi)..."
    sleep 3
done

echo ""
echo "🏁 =============================================="
echo "🏁 STATISTICHE FINALI"
echo "🏁 =============================================="
echo "📊 Totale rosbag: ${#ROSBAG_DIRS[@]}"
echo "✅ Processate con successo: $TOTAL_PROCESSED"
echo "❌ Fallite: $TOTAL_FAILED"
echo "📈 Tasso di successo: $(( TOTAL_PROCESSED * 100 / ${#ROSBAG_DIRS[@]} ))%"

if [ ${#PROCESSED_BAGS[@]} -gt 0 ]; then
    echo ""
    echo "✅ ROSBAG PROCESSATE CON SUCCESSO:"
    for bag in "${PROCESSED_BAGS[@]}"; do
        echo "   - $bag"
    done
fi

if [ ${#FAILED_BAGS[@]} -gt 0 ]; then
    echo ""
    echo "❌ ROSBAG FALLITE:"
    for bag in "${FAILED_BAGS[@]}"; do
        echo "   - $bag"
    done
fi

echo ""
echo "🎉 Processamento batch completato!"
echo "📁 Rosbag processate salvate con suffisso '_processed'"
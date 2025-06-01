import Foundation
import HealthKit
import WatchConnectivity

class HealthKitManager: ObservableObject {
    private let healthStore = HKHealthStore()
    private var webSocketTask: URLSessionWebSocketTask?
    
    // Observable properties
    @Published var heartRate: Double = 0
    @Published var isConnected: Bool = false
    
    init() {
        requestHealthKitPermissions()
        setupWebSocket()
    }
    
    private func setupWebSocket() {
        let url = URL(string: "ws://localhost:8501/ws")!
        webSocketTask = URLSession.shared.webSocketTask(with: url)
        webSocketTask?.resume()
        isConnected = true
    }
    
    private func requestHealthKitPermissions() {
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
            HKObjectType.quantityType(forIdentifier: .respiratoryRate)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            if success {
                self.startHeartRateMonitoring()
            }
        }
    }
    
    private func startHeartRateMonitoring() {
        let heartRateType = HKObjectType.quantityType(forIdentifier: .heartRate)!
        let query = HKAnchoredObjectQuery(type: heartRateType, predicate: nil, anchor: nil, limit: HKObjectQueryNoLimit) { query, samples, deletedObjects, anchor, error in
            self.processHeartRateSamples(samples)
        }
        healthStore.execute(query)
    }
    
    private func processHeartRateSamples(_ samples: [HKSample]?) {
        guard let samples = samples as? [HKQuantitySample] else { return }
        
        for sample in samples {
            let heartRate = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            DispatchQueue.main.async {
                self.heartRate = heartRate
                self.sendDataToStreamlit(heartRate: heartRate)
            }
        }
    }
    
    private func sendDataToStreamlit(heartRate: Double) {
        let data: [String: Any] = [
            "heart_rate": heartRate,
            "timestamp": Date().ISO8601Format()
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: data),
              let message = String(data: jsonData, encoding: .utf8) else { return }
        
        let wsMessage = URLSessionWebSocketTask.Message.string(message)
        webSocketTask?.send(wsMessage) { error in
            if let error = error {
                print("WebSocket error: \(error)")
            }
        }
    }
}
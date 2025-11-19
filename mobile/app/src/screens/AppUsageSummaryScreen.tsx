import React from "react";
import { View, Text, FlatList } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { RootStackParamList } from "../App";
import { useAppStore } from "@state/store";

export type AppUsageSummaryProps = NativeStackScreenProps<RootStackParamList, "AppUsageSummary">;

const AppUsageSummaryScreen: React.FC<AppUsageSummaryProps> = () => {
  const dailySummary = useAppStore((s) => s.dailySummary);

  const topApps = dailySummary?.topApps ?? [];

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 20, fontWeight: "bold", marginBottom: 12 }}>App Usage Summary</Text>
      <Text>Total screen time: {dailySummary?.totalScreenTime ?? 0} min</Text>
      <Text>Sessions: {dailySummary?.sessionCount ?? 0}</Text>

      <Text style={{ marginTop: 16, fontWeight: "bold" }}>Top apps today</Text>
      <FlatList
        data={topApps}
        keyExtractor={(item, index) => `${item}-${index}`}
        renderItem={({ item, index }) => (
          <Text>
            {index + 1}. {item}
          </Text>
        )}
      />
    </View>
  );
};

export default AppUsageSummaryScreen;

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property var theme

    Rectangle {
        anchors.fill: parent
        color: theme.surface
        radius: theme.cornerRadius
        border.color: theme.border
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: theme.padding
        spacing: theme.spacing

        Label {
            text: "Normal Preview"
            color: theme.textPrimary
            font.pixelSize: 18
            Layout.alignment: Qt.AlignLeft
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: theme.surfaceAlt
            radius: theme.cornerRadius
            border.color: theme.border

            Label {
                anchors.centerIn: parent
                text: "Normal map preview"
                color: theme.textMuted
            }
        }
    }
}
